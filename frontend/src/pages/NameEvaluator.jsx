import React, { useState } from "react";
import axios from "axios";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

const API = "/api";

const priorityLabels = {
  1: "First preference",
  2: "Second preference",
  3: "Third preference",
};

function decisionColor(label) {
  switch (label) {
    case "high":
      return "bg-emerald-600 text-emerald-50";
    case "medium":
      return "bg-amber-500 text-amber-50";
    default:
      return "bg-rose-500 text-rose-50";
  }
}

export function NameEvaluatorPage() {
  const [names, setNames] = useState([
    { priority: 1, name: "" },
    { priority: 2, name: "" },
    { priority: 3, name: "" },
  ]);
  const [entityType, setEntityType] = useState("private_limited");
  const [industry, setIndustry] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleNameChange = (index, value) => {
    const updated = [...names];
    updated[index].name = value;
    setNames(updated);
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResults(null);

    const payloadNames = names
      .filter((n) => n.name.trim().length > 0)
      .map((n) => ({ priority: n.priority, name: n.name.trim() }));

    if (payloadNames.length === 0) {
      setError("Please enter at least one proposed name.");
      return;
    }

    try {
      setLoading(true);
      const response = await axios.post(
        `${API}/evaluate-names`,
        {
          names: payloadNames,
          entity_type: entityType,
          industry: industry || null,
        },
        {
          timeout: 10000,
        },
      );
      setResults(response.data);
    } catch (err) {
      console.error(err);
      let msg = "Unable to evaluate names at the moment. Please try again.";
      if (err.response) {
        // Server responded with a status code outside 2xx
        msg = `Server Error (${err.response.status}): ${err.response.data?.detail || err.response.statusText}`;
      } else if (err.request) {
        // Request was made but no response received
        msg = "Network Error: No response received from server. Please check your connection.";
      } else {
        // Something happened in setting up the request
        msg = `Error: ${err.message}`;
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <div className="mx-auto flex max-w-5xl flex-col gap-8 px-6 py-10 lg:py-14">
        <header className="space-y-2">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400" data-testid="app-badge">
            Ministry of Corporate Affairs · Advisory Prototype
          </p>
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-semibold tracking-tight" data-testid="page-title">
            Company name acceptance probability
          </h1>
          <p className="max-w-2xl text-sm sm:text-base text-slate-300" data-testid="page-subtitle">
            Enter up to three proposed names in order of preference. The system will estimate the
            likelihood of approval based on MCA naming guidelines, historical patterns and
            phonetic similarity.
          </p>
        </header>

        <main className="grid gap-8 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)] items-start">
          <Card className="border-slate-800 bg-slate-900/70 backdrop-blur" data-testid="name-form-card">
            <CardHeader className="pb-4">
              <CardTitle className="text-base sm:text-lg text-slate-50" data-testid="form-title">
                Proposed names
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <form className="space-y-5" onSubmit={onSubmit} data-testid="name-evaluator-form">
                <div className="space-y-4" data-testid="names-input-group">
                  {names.map((item, index) => (
                    <div key={item.priority} className="space-y-1.5">
                      <label
                        className="flex items-baseline justify-between text-xs font-medium text-slate-300"
                        htmlFor={`name-${item.priority}`}
                      >
                        <span data-testid={`name-label-${item.priority}`}>
                          {priorityLabels[item.priority]}
                        </span>
                        <span className="text-[11px] text-slate-500">Priority {item.priority}</span>
                      </label>
                      <Input
                        id={`name-${item.priority}`}
                        data-testid={`name-input-${item.priority}`}
                        placeholder="e.g. Rising North Tech Private Limited"
                        value={item.name}
                        onChange={(e) => handleNameChange(index, e.target.value)}
                        className="h-10 rounded-full border-slate-700 bg-slate-950/60 text-sm text-slate-50 placeholder:text-slate-500 focus-visible:ring-emerald-500/60"
                        maxLength={200}
                      />
                    </div>
                  ))}
                </div>

                <div className="grid gap-4 sm:grid-cols-2" data-testid="meta-input-group">
                  <div className="space-y-1.5">
                    <label
                      className="text-xs font-medium text-slate-50"
                      htmlFor="entity-type"
                      data-testid="entity-type-label"
                    >
                      Entity type
                    </label>
                    <Select
                      value={entityType}
                      onValueChange={setEntityType}
                      data-testid="entity-type-select"
                    >
                      <SelectTrigger
                        id="entity-type"
                        className="h-10 rounded-full border-slate-700 bg-slate-950/60 text-xs sm:text-sm text-slate-50"
                      >
                        <SelectValue placeholder="Select entity type" />
                      </SelectTrigger>
                      <SelectContent className="border-slate-700 bg-slate-900">
                        <SelectItem value="private_limited" data-testid="entity-type-private-limited">
                          Private Limited Company
                        </SelectItem>
                        <SelectItem value="public_limited" data-testid="entity-type-public-limited">
                          Public Limited Company
                        </SelectItem>
                        <SelectItem value="opc_private_limited" data-testid="entity-type-opc">
                          One Person Company (OPC)
                        </SelectItem>
                        <SelectItem value="llp" data-testid="entity-type-llp">
                          Limited Liability Partnership (LLP)
                        </SelectItem>
                        <SelectItem value="section8" data-testid="entity-type-section8">
                          Section 8 / Non-profit
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-1.5">
                    <label
                      className="text-xs font-medium text-slate-50"
                      htmlFor="industry"
                      data-testid="industry-label"
                    >
                      Primary industry (optional)
                    </label>
                    <Input
                      id="industry"
                      data-testid="industry-input"
                      placeholder="e.g. Information Technology, Manufacturing"
                      value={industry}
                      onChange={(e) => setIndustry(e.target.value)}
                      className="h-10 rounded-full border-slate-700 bg-slate-950/60 text-sm text-slate-50 placeholder:text-slate-500 focus-visible:ring-emerald-500/60"
                      maxLength={100}
                    />
                  </div>
                </div>

                {error && (
                  <div
                    className="rounded-md border border-rose-500/60 bg-rose-500/10 px-3 py-2 text-xs text-rose-100"
                    data-testid="error-message"
                  >
                    {error}
                  </div>
                )}

                <div className="flex items-center justify-between gap-3 pt-2">
                  <p className="text-[11px] text-slate-500" data-testid="disclaimer-text">
                    This is an analytical prototype to assist entrepreneurs and officials. Final
                    decisions rest with MCA processing officers.
                  </p>
                  <Button
                    type="submit"
                    disabled={loading}
                    className="h-10 rounded-full bg-emerald-600 px-5 text-xs sm:text-sm font-medium text-emerald-50 hover:bg-emerald-500 disabled:opacity-60 disabled:hover:bg-emerald-600"
                    data-testid="evaluate-button"
                  >
                    {loading ? "Evaluating…" : "Evaluate names"}
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>

          <section className="space-y-4" data-testid="results-section">
            <Card className="border-slate-800 bg-slate-900/60" data-testid="results-card">
              <CardHeader className="pb-3">
                <CardTitle className="text-base sm:text-lg text-slate-50" data-testid="results-title">
                  Evaluation output
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-sm">
                {!results && (
                  <p className="text-xs sm:text-sm text-slate-400" data-testid="results-placeholder">
                    Run an evaluation to see acceptance probabilities, MCA rule signals and risk
                    explanations for each proposed name.
                  </p>
                )}

                {results && results.results && (
                  <div className="space-y-4" data-testid="results-list">
                    {results.results
                      .slice()
                      .sort((a, b) => a.priority - b.priority)
                      .map((item) => (
                        <Card
                          key={`${item.priority}-${item.name}`}
                          className="border border-slate-800 bg-slate-950/60"
                          data-testid={`result-card-${item.priority}`}
                        >
                          <CardContent className="space-y-3 pt-4">
                            <div className="flex flex-wrap items-start justify-between gap-2">
                              <div className="space-y-1">
                                <p
                                  className="text-[11px] uppercase tracking-[0.22em] text-slate-500"
                                  data-testid={`result-priority-${item.priority}`}
                                >
                                  Preference {item.priority}
                                </p>
                                <h2
                                  className="text-sm sm:text-base font-medium text-slate-50"
                                  data-testid={`result-name-${item.priority}`}
                                >
                                  {item.name}
                                </h2>
                              </div>
                              <Badge
                                className={`rounded-full px-3 py-1 text-[11px] font-medium ${decisionColor(item.decision_label)}`}
                                data-testid={`result-decision-badge-${item.priority}`}
                              >
                                {item.decision_label === "high"
                                  ? "High likelihood"
                                  : item.decision_label === "medium"
                                    ? "Moderate likelihood"
                                    : "Low likelihood"}
                              </Badge>
                            </div>

                            <div className="space-y-1" data-testid={`result-score-group-${item.priority}`}>
                              <div className="flex items-center justify_between text-[11px] text-slate-400">
                                <span>Estimated acceptance probability</span>
                                <span
                                  className="font-medium text-slate-100"
                                  data-testid={`result-score-text-${item.priority}`}
                                >
                                  {item.acceptance_probability.toFixed(2)}%
                                </span>
                              </div>
                              <Progress
                                value={item.acceptance_probability}
                                className="h-1.5 overflow-hidden rounded-full bg-slate-800"
                                data-testid={`result-score-bar-${item.priority}`}
                              />
                            </div>

                            {item.rule_flags && item.rule_flags.length > 0 && (
                              <div
                                className="space-y-1.5"
                                data-testid={`result-rule-flags-${item.priority}`}
                              >
                                <p className="text-[11px] font-medium text-slate-300">
                                  Key MCA rule signals
                                </p>
                                <ul className="space-y-1.5 text-[11px] text-slate-400">
                                  {item.rule_flags.map((flag, idx) => (
                                    <li
                                      key={`${flag.code}-${idx}`}
                                      className="flex items-start gap-2"
                                      data-testid={`result-flag-${item.priority}-${idx}`}
                                    >
                                      <span className="mt-[3px] h-1.5 w-1.5 rounded-full bg-slate-500" />
                                      <span>{flag.description}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {item.explanations && item.explanations.length > 0 && (
                              <div
                                className="space-y-1.5"
                                data-testid={`result-explanations-${item.priority}`}
                              >
                                <p className="text-[11px] font-medium text-slate-300">
                                  How this estimate was derived
                                </p>
                                <ul className="space-y-1.5 text-[11px] text-slate-400">
                                  {item.explanations.map((note, idx) => (
                                    <li
                                      key={idx}
                                      className="flex items-start gap-2"
                                      data-testid={`result-explanation-${item.priority}-${idx}`}
                                    >
                                      <span className="mt-[3px] h-1.5 w-1.5 rounded-full bg-emerald-500" />
                                      <span>{note}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </section>
        </main>
      </div>
    </div>
  );
}

export default NameEvaluatorPage;
