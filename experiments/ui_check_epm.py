"""Click through the plus-maze part of the demo UI in headless Chromium; screenshots in outputs/ui_check/.

Needs the server running and EPM_17 seeded (scripts/seed_ui_state.py). Saved setups are restored afterwards.
    .venv\\Scripts\\python experiments\\ui_check_epm.py
"""
import glob
import os

from playwright.sync_api import sync_playwright

URL, VIDEO, OUT = "http://127.0.0.1:8765/", "data/sturman/epm/EPM_17.mp4", "outputs/ui_check"
os.makedirs(OUT, exist_ok=True)
backup = {p: open(p, "rb").read() for p in glob.glob("outputs/ui/*/setup.json")}
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def wait_results(page):
    page.wait_for_function("() => !document.getElementById('results').classList.contains('hidden')"
                           " && document.getElementById('auto-times').innerText.startsWith('Detected')", timeout=30000)
    page.wait_for_timeout(300)


def open_time(page):
    rows = page.locator("#zones tr")
    header = [h.inner_text() for h in rows.first.locator("th").all()]
    return rows.nth(1).locator("td").nth(header.index("Open arms")).inner_text()


try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000}, accept_downloads=True)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(URL)
        page.wait_for_function("() => document.getElementById('video').options.length > 0")
        page.select_option("#video", VIDEO)
        wait_results(page)

        check("test selector shows the plus maze", page.input_value("#test") == "epm")
        check("maze settings shown, open-field settings hidden",
              page.is_visible("#epm-settings") and not page.is_visible("#of-settings"))
        n12 = page.evaluate("() => handles().length")
        check("12-corner mode: 12 handles", n12 == 12, str(n12))
        check("notice names the maze points", "Maze points" in page.inner_text("#notice"), page.inner_text("#notice")[:90])
        page.screenshot(path=f"{OUT}/epm_01_setup_12corners.png")

        page.click("button[data-tab=results]")
        header = [h.inner_text() for h in page.locator("#zones tr").first.locator("th").all()]
        check("zone table has open / closed / centre and each arm", header[:4] == ["Measure", "Open arms", "Closed arms", "Centre"],
              str(header))
        whole = page.inner_text("#whole")
        check("main indices shown", "Time in open arms" in whole and "Open-arm entries" in whole and "Total arm entries" in whole)
        page.screenshot(path=f"{OUT}/epm_02_results.png", full_page=True)
        base_open = open_time(page)

        page.click("button[data-tab=setup]")
        page.check("input[name=maze_mode][value=tips]")
        page.wait_for_timeout(1500)
        n4 = page.evaluate("() => handles().length")
        labels = page.evaluate("() => handles().map(h => h.label)")
        check("4-tip mode: handles C1 O1 C2 O2", n4 == 4 and labels == ["C1", "O1", "C2", "O2"], str(labels))
        page.screenshot(path=f"{OUT}/epm_03_setup_tips.png")

        # widen the arms: zones change, results recalculate
        page.click("button[data-tab=results]")
        page.click("button[data-tab=setup]")
        before = page.evaluate("() => JSON.stringify(geom.zones.open)")
        page.fill("#arm_width_cm", "8")
        page.dispatch_event("#arm_width_cm", "change")
        page.wait_for_function("b => JSON.stringify(geom.zones.open) !== b", arg=before, timeout=15000)
        wait_results(page)
        page.click("button[data-tab=results]")
        page.wait_for_function("b => { const r = document.querySelectorAll('#zones tr'); const h = [...r[0].children].map(c => c.innerText);"
                               " return r[1].children[h.indexOf('Open arms')].innerText !== b; }", arg=base_open, timeout=20000)
        check("changing arm width recalculates open-arm time", open_time(page) != base_open,
              f"{base_open} s -> {open_time(page)} s")
        page.click("button[data-tab=setup]")
        page.screenshot(path=f"{OUT}/epm_04_wide_arms.png")

        # drag tip C1 up by 20 px
        before = page.evaluate("() => setup.maze_tips_px[0].slice()")
        h = page.evaluate("""() => { const r = document.getElementById('canvas').getBoundingClientRect(), cv = document.getElementById('canvas');
            const [x, y] = setup.maze_tips_px[0]; return [r.left + x * scale * r.width / cv.width, r.top + y * scale * r.height / cv.height]; }""")
        page.mouse.move(h[0], h[1]); page.mouse.down(); page.mouse.move(h[0], h[1] - 20, steps=5); page.mouse.up()
        page.wait_for_timeout(1000)
        after = page.evaluate("() => setup.maze_tips_px[0]")
        check("dragging a tip moves it", abs(after[1] - before[1]) > 5, f"{[round(v) for v in before]} -> {[round(v) for v in after]}")

        page.click("button[data-tab=results]")
        with page.expect_download() as dl:
            page.click("#dl-pdf")
        path = os.path.join(OUT, dl.value.suggested_filename)
        dl.value.save_as(path)
        check("PDF report downloads", os.path.getsize(path) > 10000, f"{os.path.getsize(path)} bytes")

        # back to the open field video: the open-field form returns
        page.click("button[data-tab=setup]")
        page.select_option("#video", "data/ysmu/cntrol60_cage22_an1_OF1.mp4")
        wait_results(page)
        check("switching to an open-field video restores the open-field form",
              page.input_value("#test") == "open_field" and page.is_visible("#of-settings"))
        check("no JavaScript errors", not errors, "; ".join(errors[:3]))
        browser.close()
finally:
    for path, data in backup.items():
        with open(path, "wb") as fh:
            fh.write(data)
    for path in glob.glob("outputs/ui/*/setup.json"):
        if path not in backup:
            os.remove(path)
print(f"\n{sum(results)}/{len(results)} checks passed")
