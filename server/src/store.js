const SEED_TEAMS = [
  "Sluggers",
  "Mariners",
  "Cyclones",
  "Pioneers",
  "Rockets",
  "Grizzlies"
];

function createInitialState() {
  const teams = new Map();
  for (const name of SEED_TEAMS) {
    teams.set(name, { name, wins: 0, losses: 0, runsFor: 0, runsAgainst: 0 });
  }
  return { teams, games: [] };
}

let state = createInitialState();

export function resetStore() {
  state = createInitialState();
}

export function listTeams() {
  return [...state.teams.keys()];
}

function winPct(team) {
  const played = team.wins + team.losses;
  return played === 0 ? 0 : team.wins / played;
}

export function getStandings() {
  return [...state.teams.values()]
    .map((team) => ({
      name: team.name,
      wins: team.wins,
      losses: team.losses,
      runsFor: team.runsFor,
      runsAgainst: team.runsAgainst,
      runDiff: team.runsFor - team.runsAgainst,
      pct: Number(winPct(team).toFixed(3))
    }))
    .sort((a, b) => b.pct - a.pct || b.runDiff - a.runDiff || a.name.localeCompare(b.name));
}

export function getGames() {
  return [...state.games].sort((a, b) => b.playedAt - a.playedAt);
}

export class ValidationError extends Error {}

export function recordGame({ homeTeam, awayTeam, homeScore, awayScore }) {
  if (!homeTeam || !awayTeam) {
    throw new ValidationError("Both homeTeam and awayTeam are required.");
  }
  if (homeTeam === awayTeam) {
    throw new ValidationError("A team cannot play against itself.");
  }
  const home = state.teams.get(homeTeam);
  const away = state.teams.get(awayTeam);
  if (!home || !away) {
    throw new ValidationError("Unknown team supplied.");
  }
  const hs = Number(homeScore);
  const as = Number(awayScore);
  if (!Number.isInteger(hs) || !Number.isInteger(as) || hs < 0 || as < 0) {
    throw new ValidationError("Scores must be non-negative integers.");
  }
  if (hs === as) {
    throw new ValidationError("Baseball games cannot end in a tie.");
  }

  home.runsFor += hs;
  home.runsAgainst += as;
  away.runsFor += as;
  away.runsAgainst += hs;

  const homeWon = hs > as;
  if (homeWon) {
    home.wins += 1;
    away.losses += 1;
  } else {
    away.wins += 1;
    home.losses += 1;
  }

  const game = {
    id: state.games.length + 1,
    homeTeam,
    awayTeam,
    homeScore: hs,
    awayScore: as,
    winner: homeWon ? homeTeam : awayTeam,
    playedAt: Date.now()
  };
  state.games.push(game);
  return game;
}
