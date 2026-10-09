"""Click through the demo UI in headless Chromium and check each feature; screenshots go to outputs/ui_check/.

Needs the server running (run_ui.bat). Saved setups are backed up first and restored at the end.
    .venv\\Scripts\\python experiments\\ui_check.py
"""
import glob
import os
import shutil

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8765/"
OUT = "outputs/ui_check"
AN1, AN2 = "data/ysmu/cntrol60_cage22_an1_OF1.mp4", "data/ysmu/cntrol60_cage22_an2_OF1.mp4"
os.makedirs(OUT, exist_ok=True)
backup = {p: open(p, "rb").read() for p in glob.glob("outputs/ui/*/setup.json")}
for p in glob.glob("outputs/ui/cntrol60_cage22_an2_OF1_*/setup.json"):  # an2 must start from the copied an1 setup
    os.remove(p)
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def centre_time(page):
    return page.locator("#zones tr").nth(1).locator("td").nth(1).inner_text()


def wait_results(page, before=None):
    """Wait until the current video's analysis is shown and (optionally) the centre time changed from `before`."""
    page.wait_for_function("() => !document.getElementById('results').classList.contains('hidden')"
                           " && document.getElementById('auto-times').innerText.startsWith('Detected')", timeout=20000)
    if before is not None:
        page.wait_for_function(
            "b => { const c = document.querySelectorAll('#zones tr')[1]; return c && c.children[1].innerText !== b; }",
            arg=before, timeout=15000)
    page.wait_for_timeout(300)


try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 950}, accept_downloads=True)
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        # 1. Setup page, an1: corners from its own saved setup
        page.goto(URL)
        page.select_option("#video", AN1)
        page.wait_for_timeout(2500)
        wait_results(page)
        notice = page.inner_text("#notice")
        check("an1 notice: corners from saved setup of this video", "saved setup of this video" in notice, notice)
        page.screenshot(path=f"{OUT}/01_setup_an1.png")

        # 2. an2: corners copied from an1 (same folder/camera); automatic start ~14.9 s
        page.select_option("#video", AN2)
        page.wait_for_timeout(2500)
        wait_results(page)
        notice = page.inner_text("#notice")
        check("an2 notice: corners copied from an1", "copied from the saved setup of cntrol60_cage22_an1_OF1" in notice, notice)
        auto = page.inner_text("#auto-times")
        check("an2 automatic start ~14.9 s and end detected", "start 14.87" in auto, auto)
        page.screenshot(path=f"{OUT}/02_setup_an2.png")
        api = page.evaluate("""async () => (await (await fetch('/api/analyze?video=' + encodeURIComponent(video),
            {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(setup)})).json()).zone_rows[0][1]""")
        check("page numbers equal a direct API call", centre_time(page) == api, f"page {centre_time(page)} s, API {api} s")

        # 3. Results tab
        page.click("button[data-tab=results]")
        page.wait_for_timeout(500)
        n_whole = page.locator("#whole tr").count()
        n_zone_cols = page.locator("#zones tr").first.locator("th").count()
        imgs = page.evaluate("() => ['img-track','img-heatmap','img-per-minute'].map(i => document.getElementById(i).naturalWidth)")
        check("results: whole-test table rows", n_whole >= 10, f"{n_whole} rows")
        check("results: zone table has centre/wall/corners", n_zone_cols == 4, f"{n_zone_cols} columns")
        check("results: three images rendered", all(w > 100 for w in imgs), str(imgs))
        page.screenshot(path=f"{OUT}/03_results_an2.png", full_page=True)
        base_centre = centre_time(page)

        # 4. Centre zone 30% -> results update
        page.click("button[data-tab=setup]")
        page.fill("#center_percent", "30")
        page.dispatch_event("#center_percent", "change")
        wait_results(page, base_centre)
        c30 = centre_time(page)
        check("centre 30% recalculates", c30 != base_centre, f"{base_centre} s -> {c30} s")
        page.screenshot(path=f"{OUT}/04_centre_30pct.png")

        # 5. Centre as 5 cm from walls
        page.check("input[name=center_mode][value=cm]")
        page.fill("#center_cm", "5")
        page.dispatch_event("#center_cm", "change")
        wait_results(page, c30)
        c5 = centre_time(page)
        check("centre 5 cm from walls recalculates", c5 != c30, f"{c30} s -> {c5} s")
        page.screenshot(path=f"{OUT}/05_centre_5cm.png")

        # 6. Reset, then click the bottom-left grid square to add it to the centre
        page.click("#reset-zones")
        wait_results(page, c5)
        reset_c = centre_time(page)
        check("reset to default 50% gives the original centre time", reset_c == base_centre, f"{reset_c} s")
        pt = page.evaluate("""() => {
            const c = geom.cells.find(c => c.col === 0 && c.row === geom.cells.reduce((m, x) => Math.max(m, x.row), 0));
            const cx = c.poly.reduce((s, p) => s + p[0], 0) / 4, cy = c.poly.reduce((s, p) => s + p[1], 0) / 4;
            const r = document.getElementById('canvas').getBoundingClientRect(), cv = document.getElementById('canvas');
            return [r.left + cx * scale * r.width / cv.width, r.top + cy * scale * r.height / cv.height];
        }""")
        page.mouse.click(pt[0], pt[1])
        wait_results(page, reset_c)
        toggled = page.evaluate("() => setup.toggled_cells")
        cclick = centre_time(page)
        check("clicking a grid square toggles it into the centre", len(toggled) == 1 and cclick != reset_c,
              f"toggled {toggled}, centre {reset_c} s -> {cclick} s")
        page.screenshot(path=f"{OUT}/06_square_clicked.png")
        page.mouse.click(pt[0], pt[1])  # click again: removed
        wait_results(page, cclick)
        check("clicking it again removes it", page.evaluate("() => setup.toggled_cells.length") == 0, centre_time(page))

        # 7. Drag the top-left corner handle 25 px right/down
        before = page.evaluate("() => setup.corners_px[0].slice()")
        h = page.evaluate("""() => { const r = document.getElementById('canvas').getBoundingClientRect(), cv = document.getElementById('canvas');
            const [x, y] = setup.corners_px[0]; return [r.left + x * scale * r.width / cv.width, r.top + y * scale * r.height / cv.height]; }""")
        page.mouse.move(h[0], h[1])
        page.mouse.down()
        page.mouse.move(h[0] + 25, h[1] + 25, steps=5)
        page.mouse.up()
        page.wait_for_timeout(1500)
        after = page.evaluate("() => setup.corners_px[0]")
        moved = abs(after[0] - before[0]) > 5 and abs(after[1] - before[1]) > 5
        check("dragging a corner moves it", moved, f"{[round(v) for v in before]} -> {[round(v) for v in after]}")
        page.screenshot(path=f"{OUT}/07_corner_dragged.png")

        # 8. Timeline: set start at 60 s, then back to auto
        page.evaluate("() => { const s = document.getElementById('scrub'); s.value = 60; s.dispatchEvent(new Event('input')); }")
        page.wait_for_timeout(800)
        page.click("#set-start")
        page.wait_for_function("() => document.querySelector('#whole tr td:nth-child(2)').innerText.startsWith('60.0')", timeout=15000)
        check("set start at 60 s via timeline", True, page.locator("#whole tr td").nth(1).inner_text())
        page.screenshot(path=f"{OUT}/08_start_60s.png")
        page.click("button[data-auto=start_s]")
        page.wait_for_function("() => document.querySelector('#whole tr td:nth-child(2)').innerText.startsWith('14.9')", timeout=15000)
        check("auto start restored", True, page.locator("#whole tr td").nth(1).inner_text())

        # 9. Downloads
        page.click("button[data-tab=results]")
        for link, ext in [("#dl-csv", ".csv"), ("#dl-track", ".csv"), ("#dl-pdf", ".pdf")]:
            with page.expect_download() as dl:
                page.click(link)
            path = os.path.join(OUT, dl.value.suggested_filename)
            dl.value.save_as(path)
            check(f"download {dl.value.suggested_filename}", path.endswith(ext) and os.path.getsize(path) > 500,
                  f"{os.path.getsize(path)} bytes")

        # 10. Validation tab
        page.click("button[data-tab=validation]")
        page.wait_for_selector("#val-table tr:nth-child(15)", timeout=20000)
        cards = page.inner_text("#val-cards")
        check("validation: cards and 14-video table", page.locator("#val-table tr").count() == 15, cards.replace("\n", " | "))
        page.screenshot(path=f"{OUT}/10_validation.png", full_page=True)

        check("no JavaScript errors in the console", not errors, "; ".join(errors[:3]))
        browser.close()
finally:
    for path, data in backup.items():  # undo the setup changes made by this test
        with open(path, "wb") as fh:
            fh.write(data)
    for path in glob.glob("outputs/ui/*/setup.json"):
        if path not in backup:
            os.remove(path)

print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} checks passed; screenshots in {OUT}/")
