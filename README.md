# Baseball-Project

A full-stack baseball standings tracker. Record game results and watch team
standings update live.

## Stack

- **Server** (`server/`): Node.js + Express REST API with an in-memory data store.
- **Client** (`client/`): React + Vite single-page app.
- Managed as npm workspaces from the repository root.

## Requirements

- Node.js >= 20
- npm >= 10

## Getting started

```bash
npm install          # installs both workspaces
npm run dev:server   # API on http://localhost:3001
npm run dev:client   # Web app on http://localhost:5173 (proxies /api to the server)
```

Open http://localhost:5173 and record a game to see the standings recalculate.

## Scripts

| Command | Description |
| --- | --- |
| `npm install` | Install all workspace dependencies |
| `npm run dev:server` | Start the API with live reload (`node --watch`) |
| `npm run dev:client` | Start the Vite dev server |
| `npm test` | Run the server test suite (`node --test`) |
| `npm run build` | Build the client for production |

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/teams` | List team names |
| `GET` | `/api/standings` | Current standings, sorted by win % |
| `GET` | `/api/games` | Recorded games, newest first |
| `POST` | `/api/games` | Record a game `{ homeTeam, awayTeam, homeScore, awayScore }` |

## Cloud Agent environment

`.cursor/environment.json` configures the Cloud Agent environment: `install`
runs `npm install`, and two terminals start the API and the web app.
