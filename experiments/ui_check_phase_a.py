"""Phase A check: ANY-maze style tabs and planned-feature placeholders (docs/phase_a_navigation_plan.md).

Needs the server running (run_ui.bat). Screenshots go to outputs/ui_check/phase_a_*.png. Changes nothing on disk.
    .venv\\Scripts\\python experiments\\ui_check_phase_a.py
"""
import os

from playwright.sync_api import sync_playwright

URL, OUT = "http://127.0.0.1:8765/", "outputs/ui_check"
TABS = ["protocol", "experiment", "tests", "results", "data", "validation", "roadmap"]
os.makedirs(OUT, exist_ok=True)
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 950})
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL)
    page.evaluate("() => localStorage.removeItem('showPlanned')")
    page.reload()
    page.wait_for_function("() => typeof setup !== 'undefined' && setup !== null", timeout=20000)
    page.wait_for_timeout(1500)

    # 1. Tabs in order, each opens its page
    order = page.eval_on_selector_all("header nav .tab", "bs => bs.map(b => b.dataset.tab)")
    check("7 tabs in ANY-maze order, then Validation and Roadmap", order == TABS, str(order))
    for t in TABS:
        page.click(f"button[data-tab={t}]")
        page.wait_for_timeout(400 if t != "validation" else 2500)
        active = page.eval_on_selector_all(".tab-page.active", "ps => ps.map(p => p.id)")
        check(f"tab {t} opens its page", active == [f"tab-{t}"], str(active))
        page.screenshot(path=f"{OUT}/phase_a_{t}.png", full_page=True)

    # 2. Every planned feature appears exactly once, every slot is used, the roadmap lists all
    planned = page.evaluate("() => PLANNED.map(f => ({id: f.id, slot: f.slot}))")
    counts = page.evaluate("""() => Object.fromEntries(PLANNED.map(f => [f.id,
        document.querySelectorAll(`[data-planned-slot] [data-planned-id="${f.id}"]`).length]))""")
    bad = {k: v for k, v in counts.items() if v != 1}
    check(f"each of the {len(planned)} planned features appears exactly once", not bad, str(bad))
    slots = page.eval_on_selector_all("[data-planned-slot]", "els => els.map(e => e.dataset.plannedSlot)")
    unused = [s for s in slots if not any(f["slot"] == s for f in planned)]
    missing = sorted({f["slot"] for f in planned} - set(slots))
    check("every slot on the page has a feature, every feature has a slot", not unused and not missing,
          f"unused {unused}, missing {missing}")
    in_roadmap = page.eval_on_selector_all("#roadmap tr[data-planned-id]", "rs => rs.map(r => r.dataset.plannedId)")
    check("roadmap lists every planned feature", sorted(in_roadmap) == sorted(f["id"] for f in planned),
          f"{len(in_roadmap)} rows")
    phases = page.evaluate("() => PLANNED.every(f => f.phase in PHASES)")
    check("every feature has a known phase", phases)

    # 3. Placeholders have no working controls; clicking one changes nothing and calls no server
    page.click("button[data-tab=protocol]")
    live = page.evaluate("""() => [...document.querySelectorAll('.planned, .preview')].flatMap(p =>
        [...p.querySelectorAll('button, input, select, textarea, a')].filter(c => !c.disabled)).length""")
    check("no enabled controls inside placeholders or previews", live == 0, f"{live} enabled")
    before = page.evaluate("() => JSON.stringify(setup)")
    requests = []
    page.on("request", lambda r: requests.append(r.url) if "/api/" in r.url else None)
    for pid in ["named_zones", "stages", "live_camera", "keys"]:
        page.click(f".planned[data-planned-id={pid}]")
    page.wait_for_timeout(800)
    opened = page.locator(".planned.open").count()
    after = page.evaluate("() => JSON.stringify(setup)")
    check("clicking placeholders expands their description", opened == 4, f"{opened} open")
    check("clicking placeholders changes no setup and sends no request", before == after and not requests,
          str(requests))
    page.screenshot(path=f"{OUT}/phase_a_protocol_expanded.png", full_page=True)
    for pid in ["named_zones", "stages", "live_camera", "keys"]:
        page.click(f".planned[data-planned-id={pid}]")

    # 4. Planned test types are disabled options
    opts = page.eval_on_selector_all("#test option", "os => os.map(o => [o.value || o.textContent, o.disabled])")
    enabled = [o for o, d in opts if not d]
    check("Test dropdown: only open field and plus maze are selectable, 6 planned types disabled",
          enabled == ["open_field", "epm"] and len(opts) == 8, str(opts))

    # 5. Tests tab lists every video; clicking opens results (analysed) or protocol (not run)
    page.click("button[data-tab=tests]")
    page.wait_for_selector("#tests-table tr:nth-child(2)", timeout=10000)
    vids = page.evaluate("async () => (await (await fetch('/api/videos')).json())")
    rows = page.locator("#tests-table tr").count() - 1
    check("tests table lists every video", rows == len(vids), f"{rows} rows, {len(vids)} videos")
    done = next((v["video"] for v in vids if v["analysed"]), None)
    if done:
        page.click(f"#tests-table tr[data-video='{done}']")
        page.wait_for_function("v => video === v && document.getElementById('auto-times').innerText.startsWith('Detected')",
                               arg=done, timeout=30000)
        page.wait_for_timeout(500)
        active = page.eval_on_selector(".tab-page.active", "p => p.id")
        title = page.inner_text("#results-title")
        check("clicking an analysed test opens its results", active == "tab-results" and title == done.split("/")[-1],
              f"{active}, {title}")
        page.screenshot(path=f"{OUT}/phase_a_test_opened.png")
    todo = next((v["video"] for v in vids if not v["analysed"]), None)
    if todo:
        page.click("button[data-tab=tests]")
        page.wait_for_selector(f"#tests-table tr[data-video='{todo}']")
        page.click(f"#tests-table tr[data-video='{todo}']")
        page.wait_for_function("v => video === v", arg=todo, timeout=30000)
        page.wait_for_timeout(500)
        active = page.eval_on_selector(".tab-page.active", "p => p.id")
        check("clicking a test not run yet opens the protocol", active == "tab-protocol", active)
    else:
        check("clicking a test not run yet opens the protocol", True, "skipped: every video is analysed")

    # 6. Show planned features off hides everything planned, survives a reload
    page.click("button[data-tab=roadmap]")
    page.uncheck("#show-planned")
    page.wait_for_timeout(300)
    visible = page.evaluate("""() => [...document.querySelectorAll('.planned, .planned-tab, .planned-ctl, .preview')]
        .filter(e => e.offsetParent !== null).length""")
    active = page.eval_on_selector(".tab-page.active", "p => p.id")
    check("switch off: no placeholder visible, and the hidden Roadmap tab falls back to Protocol",
          visible == 0 and active == "tab-protocol", f"{visible} visible, {active}")
    page.reload()
    page.wait_for_timeout(1500)
    check("switch off survives a reload", not page.is_checked("#show-planned")
          and page.evaluate("() => document.body.classList.contains('hide-planned')"))
    page.screenshot(path=f"{OUT}/phase_a_protocol_clean.png", full_page=True)
    page.check("#show-planned")
    page.wait_for_timeout(300)
    check("switch on again shows placeholders", page.locator(".planned").first.is_visible())

    # 7. Narrow window: the header and panel still fit
    page.set_viewport_size({"width": 1280, "height": 800})
    page.wait_for_timeout(300)
    overflow = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
    check("no horizontal scroll at 1280 px", overflow <= 0, f"{overflow} px")
    page.screenshot(path=f"{OUT}/phase_a_1280.png")

    check("no JavaScript errors in the console", not errors, "; ".join(errors[:3]))
    browser.close()

print(f"\n{sum(results)}/{len(results)} checks passed; screenshots in {OUT}/")
