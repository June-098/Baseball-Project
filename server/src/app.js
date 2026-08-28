import express from "express";
import cors from "cors";
import {
  getGames,
  getStandings,
  listTeams,
  recordGame,
  ValidationError
} from "./store.js";

export function createApp() {
  const app = express();
  app.use(cors());
  app.use(express.json());

  app.get("/api/health", (req, res) => {
    res.json({ status: "ok" });
  });

  app.get("/api/teams", (req, res) => {
    res.json({ teams: listTeams() });
  });

  app.get("/api/standings", (req, res) => {
    res.json({ standings: getStandings() });
  });

  app.get("/api/games", (req, res) => {
    res.json({ games: getGames() });
  });

  app.post("/api/games", (req, res) => {
    try {
      const game = recordGame(req.body ?? {});
      res.status(201).json({ game, standings: getStandings() });
    } catch (err) {
      if (err instanceof ValidationError) {
        res.status(400).json({ error: err.message });
        return;
      }
      throw err;
    }
  });

  return app;
}
