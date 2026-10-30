// 한 세션에서: 캔버스 비우기 → 플러그인 워크플로우 놓기 → 예제 카드를 소스 노드에 끌어 놓기 → 노드 실행 → 캡처.
// 사용: node scene.js <board url> <workflow title> <card index a> <card index b> <shot prefix> [run: tween|inbetween|both] [lang]
const fs = require("fs");
const { chromium } = require("E:/Programming/PromBoard/node_modules/playwright");
const OUT = "E:/pb-iso/pack/work/shots";
const [url, wfTitle, ca, cb, prefix, run = "first", lang = "ko"] = process.argv.slice(2);

async function boxes(page) {
  return page.evaluate(() =>
    [...document.querySelectorAll("[data-node-id]")].map((el) => {
      const r = el.getBoundingClientRect();
      return { id: el.getAttribute("data-node-id"), x: r.x, y: r.y, w: r.width, h: r.height, text: el.innerText.slice(0, 60) };
    })
  );
}

async function drag(page, a, b) {
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(b.x, b.y, { steps: 20 });
  await page.mouse.up();
}

(async () => {
  const ctx = await chromium.launchPersistentContext(lang === "en" ? "E:/pb-iso/pack/pw-profile-en" : "E:/pb-iso/pack/pw-profile", {
    headless: true, viewport: { width: 1600, height: 900 }, locale: lang === "en" ? "en-US" : "ko-KR",
  });
  const page = ctx.pages()[0] || (await ctx.newPage());
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(5000);
  // 캔버스 비우기
  await page.mouse.click(1300, 820);
  await page.keyboard.press("Control+a");
  await page.keyboard.press("Delete");
  await page.waitForTimeout(1200);
  const dlg = page.locator("[role=alertdialog], [role=dialog]");
  if (await dlg.count()) {
    console.log("dialog", (await dlg.first().innerText()).split("\n").join(" | ").slice(0, 200));
    await page.screenshot({ path: OUT + "/" + prefix + "-dialog.png" });
    const ok = dlg.first().locator("button").filter({ hasText: /삭제|지우기|확인|Delete|Remove|OK/ });
    if (await ok.count()) await ok.last().click();
    await page.waitForTimeout(1200);
  }
  const tabs = page.locator("button", { hasText: lang === "en" ? "Workflows" : "워크플로우" });
  await tabs.nth(await tabs.count() > 1 ? 1 : 0).click().catch(async () => await tabs.first().click());
  await page.waitForTimeout(1000);
  await page.locator("button", { hasText: lang === "en" ? "Plugins" : "플러그인" }).first().click();
  await page.waitForTimeout(1200);
  await page.locator(`[title='${wfTitle}']`).first().dblclick();
  await page.waitForTimeout(3000);
  // 왼쪽 패널 밑에 깔린 노드가 보이게 캔버스를 오른쪽으로 민다(손 도구)
  await page.mouse.click(306, 47);
  await page.waitForTimeout(300);
  await drag(page, { x: 1150, y: 860 }, { x: 1150 + (+process.env.PANX || 300), y: 860 + (+process.env.PANY || 0) });
  await page.waitForTimeout(500);
  await page.mouse.click(267, 47);
  await page.waitForTimeout(500);
  await page.locator("button", { hasText: lang === "en" ? "Assets" : "에셋" }).first().click();
  await page.waitForTimeout(800);
  await page.locator("button", { hasText: lang === "en" ? "Plugins" : "플러그인" }).first().click();
  await page.waitForTimeout(1500);
  let nodes = await boxes(page);
  console.log(JSON.stringify(nodes.map((n) => [n.id, Math.round(n.x), Math.round(n.y), Math.round(n.w), Math.round(n.h), n.text.split("\n")[0]])));
  const sources = nodes.filter((n) => /소스|Source/.test(n.text)).sort((p, q) => p.y - q.y);
  const plugins = nodes.filter((n) => /2D 애니메이션/.test(n.text)).sort((p, q) => p.x - q.x);
  const imgs = page.locator("aside img, [class*=panel] img").filter({ hasNot: page.locator("xx") });
  const cards = await page.evaluate(() =>
    [...document.querySelectorAll("img")].map((im) => { const r = im.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2, w: r.width, src: im.src.slice(-40) }; })
      .filter((c) => c.x < 340 && c.y > 200 && c.w > 60)
  );
  console.log("cards", JSON.stringify(cards));
  await drag(page, cards[+ca], { x: sources[0].x + sources[0].w / 2, y: sources[0].y + sources[0].h / 2 });
  await page.waitForTimeout(2000);
  await drag(page, cards[+cb], { x: sources[1].x + sources[1].w / 2, y: sources[1].y + sources[1].h / 2 });
  await page.waitForTimeout(2500);
  const targets = run === "both" ? plugins : plugins.slice(0, 1);
  for (const [i, pn] of targets.entries()) {
    await page.mouse.click(pn.x + pn.w / 2, pn.y + 40);
    await page.waitForTimeout(1200);
    const runBtn = page.locator(`[data-node-id='${pn.id}'] button[aria-label]`);
    const labels = await runBtn.evaluateAll((els) => els.map((e) => e.getAttribute("aria-label")));
    console.log("buttons", JSON.stringify(labels));
    const go = page.locator(`[data-node-id='${pn.id}'] button`).filter({ has: page.locator("svg") });
    const label = labels.find((l) => /실행|Run|생성|Generate/.test(l));
    if (label) await page.locator(`[data-node-id='${pn.id}'] button[aria-label='${label}']`).first().click();
    else console.log("no run button");
    await page.screenshot({ path: `${OUT}/${prefix}-running-${i}.png` });
    // 결과가 날 때까지(최대 6분)
    const t0 = Date.now();
    await page.waitForTimeout(4000);
    while (Date.now() - t0 < 360000) {
      await page.waitForTimeout(3000);
      const el = page.locator(`[data-node-id='${pn.id}']`);
      const txt = (await el.innerText()).split(String.fromCharCode(10)).join(" / ").slice(0, 120);
      const vids = await el.locator("video").count();
      const busy = await el.locator("[aria-busy=true], [role=progressbar]").count();
      console.log(Math.round((Date.now() - t0) / 1000), "s", "videos", vids, "busy", busy, txt);
      if (vids > 0 && busy === 0) break;
    }
    console.log("node done in", Math.round((Date.now() - t0) / 1000), "s");
    await page.waitForTimeout(2000);
  }
  // 노드 무리를 왼쪽 패널 오른쪽에 맞춰 다시 놓는다(실행 중 화면이 밀린다)
  await page.mouse.click(1300, 860);
  await page.waitForTimeout(500);
  let now = await boxes(page);
  const minX = Math.min(...now.map((b) => b.x)), minY = Math.min(...now.map((b) => b.y));
  const panelRight = lang === "en" ? 385 : 350;
  await page.mouse.click(lang === "en" ? 334 : 306, 47);
  await page.waitForTimeout(300);
  await drag(page, { x: 1200, y: 860 }, { x: 1200 + (panelRight + 50 - minX), y: 860 + (130 - minY) });
  await page.waitForTimeout(500);
  await page.mouse.click(lang === "en" ? 295 : 267, 47);
  await page.waitForTimeout(500);
  await page.mouse.click(1550, 700);
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${OUT}/${prefix}-done.png` });
  now = await boxes(page);
  const byId = (id) => now.find((b) => b.id === id);
  const p0 = byId(plugins[0].id);
  await page.mouse.click(p0.x + p0.w / 2, p0.y + 40);
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${OUT}/${prefix}-selected.png` });
  if (plugins[1]) {
    let p1 = byId(plugins[1].id);
    if (p1.x + p1.w > 1570) {
      await page.mouse.click(lang === "en" ? 334 : 306, 47);
      await page.waitForTimeout(300);
      await drag(page, { x: 1200, y: 860 }, { x: 1200 - (p1.x + p1.w - 1560), y: 860 });
      await page.waitForTimeout(500);
      await page.mouse.click(lang === "en" ? 295 : 267, 47);
      await page.waitForTimeout(500);
      now = await boxes(page);
      p1 = now.find((b) => b.id === plugins[1].id);
    }
    await page.mouse.click(p1.x + p1.w / 2, p1.y + 40);
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `${OUT}/${prefix}-selected2.png` });
  }
  console.log(JSON.stringify((await boxes(page)).map((n) => [n.id, Math.round(n.x), Math.round(n.y), Math.round(n.w), Math.round(n.h)])));
  await ctx.close();
})();
