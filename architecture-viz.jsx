import { useState } from "react";

const approaches = [
  {
    id: 1,
    name: "The Ouroboros",
    subtitle: "Single Agent Self-Modification Loop",
    emoji: "🐍",
    color: "#e74c3c",
    bg: "#fdf2f2",
    philosophy: "One agent, one repo, continuous self-consumption. The agent writes code, reviews its own PRs using an alternate persona, then iterates.",
    loop: ["Code", "Commit", "PR", "Self-Review", "Accept/Reject", "Benchmark", "Learn"],
    mechanism: "Dual-mode system: Builder persona (creative, temp=0.8) and Reviewer persona (strict, temp=0.3) create an adversarial dynamic that drives quality upward.",
    strengths: ["Simplest to implement", "No coordination overhead", "Fast iteration cycles"],
    weaknesses: ["Limited perspective diversity", "Potential echo chamber", "May miss systemic issues"],
    chosen: true,
  },
  {
    id: 2,
    name: "The Parliament",
    subtitle: "Multi-Agent Voting System",
    emoji: "🏛️",
    color: "#3498db",
    bg: "#f0f7ff",
    philosophy: "Five specialist agents (Architect, Security, Performance, Style, Test) review every PR. A Merge Controller tallies weighted votes.",
    loop: ["Code", "PR", "Parallel Review (5 agents)", "Vote", "Merge/Reject", "Metrics"],
    mechanism: "Weighted voting based on historical accuracy. Agents that catch real bugs earn higher voting power over time. Requires consensus threshold (4/5).",
    strengths: ["Diverse perspectives", "Catches more issues", "Weighted expertise"],
    weaknesses: ["Higher token cost", "Potential deadlocks", "Complex coordination"],
    chosen: false,
  },
  {
    id: 3,
    name: "The Dojo",
    subtitle: "Benchmark-Driven Evolution",
    emoji: "🥋",
    color: "#2ecc71",
    bg: "#f0fdf4",
    philosophy: "The agent improves by competing against itself on benchmarks. It modifies its own SOPs and skills based on empirical evidence from score changes.",
    loop: ["Benchmark", "Score", "Analyze Failures", "Update SOPs/Skills", "Re-benchmark", "Compare"],
    mechanism: "Evolution log tracks which SOP/skill changes correlate with score improvements. The agent builds a model of what makes its own instructions effective.",
    strengths: ["Empirically grounded", "Measurable improvement", "Self-correcting"],
    weaknesses: ["Benchmark-gaming risk", "Narrow optimization", "Goodhart's Law"],
    chosen: true,
  },
  {
    id: 4,
    name: "The Assembly Line",
    subtitle: "Spec-Driven Pipeline",
    emoji: "🏭",
    color: "#9b59b6",
    bg: "#faf5ff",
    philosophy: "Kiro-inspired spec-driven development applied recursively. PM → Architect → Developer → QA → DevOps → Retrospective, each with its own SOP.",
    loop: ["Spec", "Design", "Implement", "Test", "Deploy", "Retrospect", "Update Pipeline"],
    mechanism: "The Retrospective Agent uses Strands Eval SOP to measure pipeline efficiency, then modifies upstream agents' SOPs after each cycle.",
    strengths: ["Production-realistic", "Comprehensive coverage", "Clear separation of concerns"],
    weaknesses: ["Complex orchestration", "Slow cycles", "Over-engineering risk"],
    chosen: true,
  },
  {
    id: 5,
    name: "The Genetic Algorithm",
    subtitle: "Population-Based Optimization",
    emoji: "🧬",
    color: "#e67e22",
    bg: "#fff8f0",
    philosophy: "Maintain N agent configurations with different prompts, tools, and strategies. Run all on same tasks. Top performers reproduce; poor ones die off.",
    loop: ["Spawn Population", "Task Assignment", "Score", "Select", "Crossover", "Mutate", "Repeat"],
    mechanism: "Configuration genes stored as markdown files. Crossover = combining sections from parent configs. Mutation = LLM-generated variations of existing instructions.",
    strengths: ["Wide solution exploration", "Avoids local optima", "Novel combinations"],
    weaknesses: ["Extremely token-intensive", "Complex infrastructure", "Slow convergence"],
    chosen: false,
  },
];

const techStack = [
  { name: "Claude Code", role: "Agent runtime + CLI", icon: "⚡", desc: "Terminal-native agentic coding with CLAUDE.md steering, hooks, subagents, and MCP" },
  { name: "Kiro IDE", role: "Spec-driven development", icon: "📋", desc: "Requirements → Design → Tasks cascade with EARS notation and property-based testing" },
  { name: "Strands Agents", role: "Agent framework", icon: "🔗", desc: "Model-driven agents with SOPs, Skills (progressive disclosure), and multi-agent patterns" },
  { name: "MCP", role: "Tool connections", icon: "🔌", desc: "Model Context Protocol for connecting GitHub, Jira, databases, and custom tooling" },
  { name: "Agent SOPs", role: "Structured workflows", icon: "📑", desc: "Markdown workflows with RFC 2119 constraints (MUST/SHOULD/MAY) for reliable execution" },
  { name: "Skills", role: "Progressive context", icon: "🎯", desc: "3-phase loading: metadata (~100 tokens) → instructions (<5K) → resources (on demand)" },
];

const systemFlow = [
  { step: 1, name: "Benchmark Baseline", agent: "Benchmarker", desc: "Run 7 benchmarks: tests, complexity, types, coverage, lint, docs, file org" },
  { step: 2, name: "Plan Improvement", agent: "Architect", desc: "Analyze scores + history → identify highest-impact change → write spec" },
  { step: 3, name: "Implement Change", agent: "Builder", desc: "Create branch → write code + tests → commit with rationale" },
  { step: 4, name: "Self-Review", agent: "Reviewer", desc: "Strict review: security, performance, correctness, style, test coverage" },
  { step: 5, name: "Validate", agent: "Benchmarker", desc: "Re-run all benchmarks → compare deltas → flag regressions" },
  { step: 6, name: "Merge Decision", agent: "Controller", desc: "Accept if: no critical findings + no regressions + tests pass" },
  { step: 7, name: "Retrospective", agent: "Retrospective", desc: "Every 5 gens: analyze patterns → update SOPs + steering → evolve process" },
];

function LoopStep({ name, index, total, color }) {
  const angle = (index / total) * 2 * Math.PI - Math.PI / 2;
  const radius = 90;
  const x = 50 + radius * Math.cos(angle) / 1.8;
  const y = 50 + radius * Math.sin(angle) / 1.8;
  return (
    <div
      style={{
        position: "absolute",
        left: `${x}%`,
        top: `${y}%`,
        transform: "translate(-50%, -50%)",
        background: color,
        color: "white",
        padding: "4px 10px",
        borderRadius: "16px",
        fontSize: "11px",
        fontWeight: 600,
        whiteSpace: "nowrap",
        boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
      }}
    >
      {name}
    </div>
  );
}

export default function App() {
  const [activeApproach, setActiveApproach] = useState(0);
  const [activeTab, setActiveTab] = useState("approaches");
  const current = approaches[activeApproach];

  return (
    <div style={{
      fontFamily: "'DM Sans', 'Segoe UI', system-ui, sans-serif",
      background: "#0f0f13",
      color: "#e8e6e3",
      minHeight: "100vh",
      padding: "24px",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "32px" }}>
        <h1 style={{
          fontSize: "28px",
          fontWeight: 700,
          background: "linear-gradient(135deg, #e74c3c, #9b59b6, #3498db)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          marginBottom: "8px",
        }}>
          Self-Improving AI Coding Agent
        </h1>
        <p style={{ color: "#8a8a8a", fontSize: "14px" }}>
          5 architectural approaches + working implementation • Claude Code × Kiro × Strands
        </p>
      </div>

      {/* Tab Bar */}
      <div style={{ display: "flex", gap: "4px", marginBottom: "24px", justifyContent: "center" }}>
        {[
          { id: "approaches", label: "5 Approaches" },
          { id: "flow", label: "System Flow" },
          { id: "tech", label: "Tech Stack" },
          { id: "code", label: "Code Structure" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: "8px 20px",
              borderRadius: "8px",
              border: "1px solid",
              borderColor: activeTab === tab.id ? "#6366f1" : "#2a2a30",
              background: activeTab === tab.id ? "#1e1b4b" : "transparent",
              color: activeTab === tab.id ? "#a5b4fc" : "#6b7280",
              cursor: "pointer",
              fontSize: "13px",
              fontWeight: 600,
              transition: "all 0.2s",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Approaches Tab */}
      {activeTab === "approaches" && (
        <div>
          {/* Approach Selector */}
          <div style={{ display: "flex", gap: "8px", marginBottom: "24px", flexWrap: "wrap", justifyContent: "center" }}>
            {approaches.map((a, i) => (
              <button
                key={a.id}
                onClick={() => setActiveApproach(i)}
                style={{
                  padding: "10px 16px",
                  borderRadius: "12px",
                  border: `2px solid ${i === activeApproach ? a.color : "#2a2a30"}`,
                  background: i === activeApproach ? a.color + "18" : "#1a1a20",
                  color: i === activeApproach ? a.color : "#8a8a8a",
                  cursor: "pointer",
                  fontSize: "13px",
                  fontWeight: 600,
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  transition: "all 0.2s",
                  position: "relative",
                }}
              >
                <span style={{ fontSize: "18px" }}>{a.emoji}</span>
                {a.name}
                {a.chosen && (
                  <span style={{
                    position: "absolute",
                    top: "-6px",
                    right: "-6px",
                    background: "#22c55e",
                    color: "white",
                    borderRadius: "50%",
                    width: "16px",
                    height: "16px",
                    fontSize: "10px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}>✓</span>
                )}
              </button>
            ))}
          </div>

          {/* Active Approach Detail */}
          <div style={{
            background: "#1a1a20",
            borderRadius: "16px",
            border: `1px solid ${current.color}30`,
            padding: "24px",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
              <span style={{ fontSize: "36px" }}>{current.emoji}</span>
              <div>
                <h2 style={{ fontSize: "22px", fontWeight: 700, color: current.color, margin: 0 }}>
                  {current.name}
                </h2>
                <p style={{ color: "#8a8a8a", fontSize: "13px", margin: 0 }}>{current.subtitle}</p>
              </div>
              {current.chosen && (
                <span style={{
                  marginLeft: "auto",
                  background: "#22c55e20",
                  color: "#22c55e",
                  padding: "4px 12px",
                  borderRadius: "20px",
                  fontSize: "12px",
                  fontWeight: 600,
                }}>✓ Used in Implementation</span>
              )}
            </div>

            <p style={{ color: "#b0b0b0", fontSize: "14px", lineHeight: 1.6, marginBottom: "20px" }}>
              {current.philosophy}
            </p>

            {/* Loop Visualization */}
            <div style={{
              position: "relative",
              height: "220px",
              marginBottom: "20px",
            }}>
              <div style={{
                position: "absolute",
                left: "50%",
                top: "50%",
                transform: "translate(-50%, -50%)",
                width: "60px",
                height: "60px",
                borderRadius: "50%",
                background: current.color + "20",
                border: `2px solid ${current.color}40`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "24px",
              }}>
                🔄
              </div>
              {current.loop.map((step, i) => (
                <LoopStep
                  key={step}
                  name={step}
                  index={i}
                  total={current.loop.length}
                  color={current.color}
                />
              ))}
            </div>

            {/* Mechanism */}
            <div style={{
              background: "#12121a",
              borderRadius: "10px",
              padding: "14px 18px",
              marginBottom: "16px",
              borderLeft: `3px solid ${current.color}`,
            }}>
              <div style={{ color: "#8a8a8a", fontSize: "11px", fontWeight: 600, textTransform: "uppercase", marginBottom: "4px" }}>
                Key Mechanism
              </div>
              <p style={{ color: "#d0d0d0", fontSize: "13px", lineHeight: 1.5, margin: 0 }}>
                {current.mechanism}
              </p>
            </div>

            {/* Strengths / Weaknesses */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
              <div style={{ background: "#0a1a0a", borderRadius: "10px", padding: "14px" }}>
                <div style={{ color: "#22c55e", fontSize: "12px", fontWeight: 600, marginBottom: "8px" }}>
                  ✦ Strengths
                </div>
                {current.strengths.map((s) => (
                  <div key={s} style={{ color: "#86efac", fontSize: "12px", marginBottom: "4px" }}>• {s}</div>
                ))}
              </div>
              <div style={{ background: "#1a0a0a", borderRadius: "10px", padding: "14px" }}>
                <div style={{ color: "#ef4444", fontSize: "12px", fontWeight: 600, marginBottom: "8px" }}>
                  ⚠ Weaknesses
                </div>
                {current.weaknesses.map((w) => (
                  <div key={w} style={{ color: "#fca5a5", fontSize: "12px", marginBottom: "4px" }}>• {w}</div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* System Flow Tab */}
      {activeTab === "flow" && (
        <div style={{ background: "#1a1a20", borderRadius: "16px", padding: "24px" }}>
          <h2 style={{ fontSize: "18px", fontWeight: 700, marginBottom: "20px", color: "#a5b4fc" }}>
            Improvement Loop — One Generation
          </h2>
          <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
            {systemFlow.map((step, i) => (
              <div key={step.step} style={{ display: "flex", alignItems: "stretch", gap: "16px" }}>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "32px" }}>
                  <div style={{
                    width: "32px",
                    height: "32px",
                    borderRadius: "50%",
                    background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
                    color: "white",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "14px",
                    fontWeight: 700,
                    flexShrink: 0,
                  }}>
                    {step.step}
                  </div>
                  {i < systemFlow.length - 1 && (
                    <div style={{ width: "2px", flex: 1, background: "#6366f140", minHeight: "20px" }} />
                  )}
                </div>
                <div style={{
                  flex: 1,
                  background: "#12121a",
                  borderRadius: "10px",
                  padding: "12px 16px",
                  marginBottom: "8px",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
                    <span style={{ fontWeight: 700, fontSize: "14px", color: "#e8e6e3" }}>{step.name}</span>
                    <span style={{
                      background: "#6366f120",
                      color: "#a5b4fc",
                      padding: "2px 8px",
                      borderRadius: "12px",
                      fontSize: "11px",
                      fontWeight: 600,
                    }}>{step.agent}</span>
                  </div>
                  <p style={{ color: "#8a8a8a", fontSize: "12px", margin: 0, lineHeight: 1.4 }}>
                    {step.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>
          <div style={{
            marginTop: "20px",
            padding: "14px",
            background: "#0a1a0a",
            borderRadius: "10px",
            borderLeft: "3px solid #22c55e",
          }}>
            <span style={{ color: "#22c55e", fontWeight: 600, fontSize: "13px" }}>
              Hybrid Design: </span>
            <span style={{ color: "#86efac", fontSize: "13px" }}>
              Ouroboros (self-modification loop) + Dojo (benchmark-driven) + Assembly Line (spec-driven pipeline)
            </span>
          </div>
        </div>
      )}

      {/* Tech Stack Tab */}
      {activeTab === "tech" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
          {techStack.map((tech) => (
            <div key={tech.name} style={{
              background: "#1a1a20",
              borderRadius: "12px",
              padding: "18px",
              border: "1px solid #2a2a30",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                <span style={{ fontSize: "22px" }}>{tech.icon}</span>
                <div>
                  <div style={{ fontWeight: 700, fontSize: "15px" }}>{tech.name}</div>
                  <div style={{ color: "#6366f1", fontSize: "11px", fontWeight: 600 }}>{tech.role}</div>
                </div>
              </div>
              <p style={{ color: "#8a8a8a", fontSize: "12px", lineHeight: 1.5, margin: 0 }}>
                {tech.desc}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Code Structure Tab */}
      {activeTab === "code" && (
        <div style={{ background: "#1a1a20", borderRadius: "16px", padding: "24px" }}>
          <h2 style={{ fontSize: "18px", fontWeight: 700, marginBottom: "16px", color: "#a5b4fc" }}>
            Project Structure
          </h2>
          <pre style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: "12px",
            lineHeight: 1.7,
            color: "#d0d0d0",
            background: "#12121a",
            borderRadius: "10px",
            padding: "18px",
            overflow: "auto",
          }}>{`self-improving-agent/
├── main.py                    # CLI: run | loop | bench | status | history | reset
├── CLAUDE.md                  # Steering document (project rules & conventions)
├── pyproject.toml             # Python project config
│
├── src/
│   ├── core/
│   │   ├── models.py          # Data models (PR, Review, Benchmark, State, Config)
│   │   ├── loop.py            # ⭐ Core improvement loop (Ouroboros engine)
│   │   └── providers.py       # LLM providers (Anthropic, Mock)
│   │
│   ├── agents/
│   │   └── personas.py        # 5 agent personas (Builder, Reviewer, Architect, etc.)
│   │
│   ├── benchmarks/
│   │   └── runner.py          # 7 benchmarks (tests, complexity, types, coverage, ...)
│   │
│   └── git_ops/
│       └── git_manager.py     # Git operations (branch, commit, PR, merge, rollback)
│
├── sops/
│   └── code-improvement.sop.md  # Strands-format SOP for the improvement cycle
│
├── skills/
│   └── self-review/
│       └── SKILL.md           # Progressive-disclosure skill for code review
│
├── steering/
│   └── project-rules.md       # Kiro-style steering rules
│
├── tests/
│   └── test_agent.py          # 24 tests (models, personas, git, benchmarks, integration)
│
└── docs/
    └── ARCHITECTURE.md        # Full research + 5 approaches documentation`}</pre>

          <div style={{ marginTop: "16px", display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "8px" }}>
            {[
              { label: "Tests Passing", value: "24/24", color: "#22c55e" },
              { label: "Agent Personas", value: "5", color: "#6366f1" },
              { label: "Benchmarks", value: "7", color: "#f59e0b" },
            ].map((stat) => (
              <div key={stat.label} style={{
                background: "#12121a",
                borderRadius: "10px",
                padding: "12px",
                textAlign: "center",
              }}>
                <div style={{ fontSize: "24px", fontWeight: 700, color: stat.color }}>{stat.value}</div>
                <div style={{ fontSize: "11px", color: "#6b7280" }}>{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
