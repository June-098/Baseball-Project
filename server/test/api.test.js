import test from "node:test";
import assert from "node:assert/strict";
import { createApp } from "../src/app.js";
import { resetStore } from "../src/store.js";

async function start() {
  resetStore();
  const server = createApp().listen(0);
  await new Promise((resolve) => server.once("listening", resolve));
  const { port } = server.address();
  const base = `http://127.0.0.1:${port}`;
  return { server, base };
}

test("health check responds ok", async () => {
  const { server, base } = await start();
  try {
    const res = await fetch(`${base}/api/health`);
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), { status: "ok" });
  } finally {
    server.close();
  }
});

test("standings start at zero wins", async () => {
  const { server, base } = await start();
  try {
    const res = await fetch(`${base}/api/standings`);
    const { standings } = await res.json();
    assert.equal(standings.length, 6);
    assert.ok(standings.every((t) => t.wins === 0 && t.losses === 0));
  } finally {
    server.close();
  }
});

test("recording a game updates standings", async () => {
  const { server, base } = await start();
  try {
    const res = await fetch(`${base}/api/games`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        homeTeam: "Sluggers",
        awayTeam: "Mariners",
        homeScore: 5,
        awayScore: 3
      })
    });
    assert.equal(res.status, 201);
    const { game, standings } = await res.json();
    assert.equal(game.winner, "Sluggers");
    const sluggers = standings.find((t) => t.name === "Sluggers");
    const mariners = standings.find((t) => t.name === "Mariners");
    assert.equal(sluggers.wins, 1);
    assert.equal(sluggers.losses, 0);
    assert.equal(sluggers.runDiff, 2);
    assert.equal(mariners.losses, 1);
    assert.equal(standings[0].name, "Sluggers");
  } finally {
    server.close();
  }
});

test("rejects tie games", async () => {
  const { server, base } = await start();
  try {
    const res = await fetch(`${base}/api/games`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        homeTeam: "Sluggers",
        awayTeam: "Mariners",
        homeScore: 4,
        awayScore: 4
      })
    });
    assert.equal(res.status, 400);
    const body = await res.json();
    assert.match(body.error, /tie/);
  } finally {
    server.close();
  }
});

test("rejects a team playing itself", async () => {
  const { server, base } = await start();
  try {
    const res = await fetch(`${base}/api/games`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        homeTeam: "Sluggers",
        awayTeam: "Sluggers",
        homeScore: 4,
        awayScore: 2
      })
    });
    assert.equal(res.status, 400);
  } finally {
    server.close();
  }
});
