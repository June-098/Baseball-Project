import { useEffect, useMemo, useState } from "react";

const EMPTY_FORM = { homeTeam: "", awayTeam: "", homeScore: "", awayScore: "" };

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

export default function App() {
  const [teams, setTeams] = useState([]);
  const [standings, setStandings] = useState([]);
  const [games, setGames] = useState([]);
  const [form, setForm] = useState(EMPTY_FORM);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  async function refresh() {
    const [{ standings }, { games }] = await Promise.all([
      getJSON("/api/standings"),
      getJSON("/api/games")
    ]);
    setStandings(standings);
    setGames(games);
  }

  useEffect(() => {
    getJSON("/api/teams").then(({ teams }) => setTeams(teams)).catch(() => {});
    refresh().catch((err) => setError(err.message));
  }, []);

  const awayOptions = useMemo(
    () => teams.filter((t) => t !== form.homeTeam),
    [teams, form.homeTeam]
  );

  function update(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  async function submit(event) {
    event.preventDefault();
    setError("");
    setStatus("");
    try {
      const res = await fetch("/api/games", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          homeTeam: form.homeTeam,
          awayTeam: form.awayTeam,
          homeScore: Number(form.homeScore),
          awayScore: Number(form.awayScore)
        })
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || "Could not record game");
      setStandings(body.standings);
      setStatus(`Recorded: ${body.game.winner} won.`);
      setForm(EMPTY_FORM);
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>⚾ Baseball Standings Tracker</h1>
        <p>Record game results and watch the standings update live.</p>
      </header>

      <main className="grid">
        <section className="card">
          <h2>Standings</h2>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Team</th>
                <th>W</th>
                <th>L</th>
                <th>PCT</th>
                <th>RD</th>
              </tr>
            </thead>
            <tbody>
              {standings.map((team, i) => (
                <tr key={team.name}>
                  <td>{i + 1}</td>
                  <td className="team">{team.name}</td>
                  <td>{team.wins}</td>
                  <td>{team.losses}</td>
                  <td>{team.pct.toFixed(3)}</td>
                  <td className={team.runDiff >= 0 ? "pos" : "neg"}>
                    {team.runDiff > 0 ? `+${team.runDiff}` : team.runDiff}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="card">
          <h2>Record a Game</h2>
          <form onSubmit={submit} className="form">
            <label>
              Home team
              <select
                value={form.homeTeam}
                onChange={(e) => update("homeTeam", e.target.value)}
                required
              >
                <option value="">Select…</option>
                {teams.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </label>
            <label>
              Home score
              <input
                type="number"
                min="0"
                value={form.homeScore}
                onChange={(e) => update("homeScore", e.target.value)}
                required
              />
            </label>
            <label>
              Away team
              <select
                value={form.awayTeam}
                onChange={(e) => update("awayTeam", e.target.value)}
                required
              >
                <option value="">Select…</option>
                {awayOptions.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </label>
            <label>
              Away score
              <input
                type="number"
                min="0"
                value={form.awayScore}
                onChange={(e) => update("awayScore", e.target.value)}
                required
              />
            </label>
            <button type="submit">Record game</button>
          </form>
          {status && <p className="status ok" role="status">{status}</p>}
          {error && <p className="status err" role="alert">{error}</p>}

          <h3>Recent games</h3>
          <ul className="games">
            {games.length === 0 && <li className="muted">No games yet.</li>}
            {games.map((g) => (
              <li key={g.id}>
                <span className={g.winner === g.homeTeam ? "win" : ""}>
                  {g.homeTeam} {g.homeScore}
                </span>
                <span className="vs">vs</span>
                <span className={g.winner === g.awayTeam ? "win" : ""}>
                  {g.awayTeam} {g.awayScore}
                </span>
              </li>
            ))}
          </ul>
        </section>
      </main>
    </div>
  );
}
