import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const imageDir = path.join(root, "public", "images", "cvor");
const dataDir = path.join(root, "public", "data", "cvor");

const methods = [
  { method: "Base / no update", id: 0.5960416667, ood: 0.49875 },
  { method: "CVoR-SFT", id: 0.636875, ood: 0.5658333333 },
  { method: "CVoR + NLL gate", id: 0.6185416667, ood: 0.54 },
  { method: "Earliest-Repair SFT", id: 0.6514583333, ood: 0.57875 },
  { method: "Random-Repair SFT", id: 0.654375, ood: 0.5841666667 },
  { method: "Uncertainty SFT", id: 0.6239583333, ood: 0.5479166667 },
  { method: "Teacher BC, equal labels", id: 0.60625, ood: 0.5241666667 },
  { method: "Sham-CVoR", id: 0.509375, ood: 0.3741666667 },
  { method: "RL, 0.25x interactions", id: 0.6683333333, ood: 0.60625 },
  { method: "RL, 0.5x interactions", id: 0.710625, ood: 0.6558333333 },
  { method: "RL, matched interactions", id: 0.7566666667, ood: 0.7179166667 },
];

const contrasts = {
  cvor_minus_base: {
    point: 0.0408333333,
    ci95: [0.024375, 0.0583333333],
    positive_seeds: 8,
  },
  cvor_minus_earliest: {
    point: -0.0145833333,
    ci95: [-0.0358385417, 0.0097916667],
    positive_seeds: 2,
    per_seed: [
      0.035, -0.02, -0.0233333333, -0.045, -0.045, -0.0266666667, -0.0316666667,
      0.04,
    ],
  },
  cvor_minus_random: {
    point: -0.0175,
    ci95: [-0.0318802083, -0.0022916667],
    positive_seeds: 1,
  },
  cvor_minus_rl_quarter: {
    point: -0.0314583333,
    ci95: [-0.04375, -0.018125],
    positive_seeds: 0,
  },
  cvor_minus_rl_matched: {
    point: -0.1197916667,
    ci95: [-0.1377083333, -0.1010416667],
    positive_seeds: 0,
  },
  sham_minus_base: {
    point: -0.0866666667,
    ci95: [-0.1106302083, -0.0597916667],
    positive_seeds: 0,
  },
};

const mechanism = {
  cvor: { audit_delta: 0.7862955729, audit_rescue: 0.7906901042 },
  earliest: { audit_delta: 0.2610677083, audit_rescue: 0.2880859375 },
  random: { audit_delta: 0.2626953125, audit_rescue: 0.2775065104 },
  uncertainty: { audit_delta: 0.1809895833, audit_rescue: 0.1883138021 },
  cvor_minus_random_rescue: {
    point: 0.5131835938,
    ci95: [0.4796549479, 0.5458984375],
  },
  selection_to_audit_spearman_mean: 0.7339281013,
};

const publicData = {
  schema_version: "cvor-blog-summary-v1",
  frozen_date: "2026-08-14",
  design: {
    checkpoint_seeds: 8,
    collection_tasks_per_seed: 900,
    id_tasks_per_seed: 600,
    ood_tasks_per_seed: 300,
    selection_suffix_seeds: 4,
    independent_audit_suffix_seeds: 8,
    independent_audit_scope: "all 7,960 candidates",
    repair_labels_per_sft_arm: 96,
    anchor_labels_per_sft_arm: 96,
    optimizer_steps_per_sft_arm: 36,
    outcome_rows: 79200,
    integrity_checks: "21/21",
    policy_observation: "36-feature local observation",
    selection_tie_break:
      "maximum selection delta, then lower repair NLL, then earlier step and stable candidate ID",
    max_delta_tie_task_groups: 1382,
    total_candidate_task_groups: 4359,
    tie_break_sensitivity_fraction: 0.133,
  },
  methods,
  contrasts,
  mechanism,
  rl: {
    closure: 0.2542153048,
    cvor_gap_to_matched_rl: -0.1197916667,
    cvor_environment_transitions_per_seed: 62449.875,
    rl_quarter_environment_transitions_per_seed: 15924.875,
    rl_matched_environment_transitions_per_seed: 62830.125,
  },
  decision: {
    mechanism: "PASS",
    downstream_training_and_rl_closure: "NO-GO",
    full_cost_pareto: "NOT EVALUABLE",
  },
};

const escapeXml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

const pct = (value) => `${(value * 100).toFixed(1)}%`;

function outcomesSvg() {
  const rows = [
    {
      label: "Base / no update",
      value: methods[0].id,
      fill: "#64748b",
      stroke: "#475569",
    },
    {
      label: "CVoR-SFT",
      value: methods[1].id,
      fill: "#0b4f8a",
      stroke: "#08375f",
    },
    {
      label: "Earliest-Repair SFT",
      value: methods[3].id,
      fill: "#9cc4df",
      stroke: "#356b92",
    },
    {
      label: "Random-Repair SFT",
      value: methods[4].id,
      fill: "#d7e7f2",
      stroke: "#356b92",
    },
    {
      label: "RL, 0.25x interactions",
      value: methods[8].id,
      fill: "#f2c18d",
      stroke: "#9a4c0b",
    },
    {
      label: "RL, 0.5x interactions",
      value: methods[9].id,
      fill: "#d97706",
      stroke: "#92400e",
    },
    {
      label: "RL, matched interactions",
      value: methods[10].id,
      fill: "#a3470a",
      stroke: "#7c2d12",
    },
  ];
  const centers = [190, 250, 310, 370, 460, 520, 580];
  const left = 320;
  const plotWidth = 800;
  const max = 0.8;
  const grid = [0, 0.2, 0.4, 0.6, 0.8]
    .map((value) => {
      const x = left + (value / max) * plotWidth;
      return `<line x1="${x}" y1="145" x2="${x}" y2="610" class="grid"/><text x="${x}" y="635" class="tick" text-anchor="middle">${Math.round(value * 100)}%</text>`;
    })
    .join("");
  const bars = rows
    .map((row, index) => {
      const width = (row.value / max) * plotWidth;
      const y = centers[index] - 17;
      return `<text x="290" y="${centers[index] + 6}" class="label" text-anchor="end">${escapeXml(row.label)}</text>
        <rect x="${left}" y="${y}" width="${width.toFixed(1)}" height="34" rx="4" fill="${row.fill}" stroke="${row.stroke}" stroke-width="1.5"/>
        <text x="${(left + width + 12).toFixed(1)}" y="${centers[index] + 6}" class="value">${pct(row.value)}</text>`;
    })
    .join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 700" role="img" aria-labelledby="title desc">
  <title id="title">Selected held-out terminal success by training arm</title>
  <desc id="desc">Horizontal bars compare Base, three supervised repair arms, and three on-policy reinforcement learning budgets. CVoR reaches 63.7 percent, while matched-interaction RL reaches 75.7 percent.</desc>
  <rect width="1200" height="700" fill="#ffffff"/>
  <style>
    text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}
    .title{font-size:28px;font-weight:700}.subtitle{font-size:16px;fill:#526071}.label{font-size:16px;font-weight:600}
    .value{font-size:16px;font-weight:700;font-variant-numeric:tabular-nums}.tick{font-size:14px;fill:#64748b}
    .grid{stroke:#e2e8f0;stroke-width:1}.group{font-size:13px;font-weight:700;letter-spacing:.08em;fill:#64748b}
    .note{font-size:14px;fill:#64748b}
  </style>
  <text x="70" y="55" class="title">Selected held-out terminal success by update arm</text>
  <text x="70" y="87" class="subtitle">Mean over 8 paired checkpoints · 600 ID tasks per checkpoint · greedy terminal success</text>
  <text x="70" y="142" class="group">SUPERVISED UPDATE</text>
  <text x="70" y="422" class="group">ON-POLICY RL</text>
  ${grid}
  <line x1="70" y1="415" x2="1140" y2="415" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="5 5"/>
  ${bars}
  <text x="70" y="674" class="note">RL uses terminal {0,1} reward. “Matched” is the first RL batch at or above CVoR’s transition target, not equal total economic cost.</text>
</svg>`;
}

function mechanismSvg() {
  const rows = [
    {
      label: "CVoR",
      rescue: 0.7906901042,
      gain: 0.0408333333,
      fill: "#0b4f8a",
      stroke: "#08375f",
    },
    {
      label: "Earliest",
      rescue: 0.2880859375,
      gain: 0.0554166667,
      fill: "#a9cce3",
      stroke: "#356b92",
    },
    {
      label: "Random",
      rescue: 0.2775065104,
      gain: 0.0583333333,
      fill: "#d7e7f2",
      stroke: "#356b92",
    },
  ];
  const centers = [240, 330, 420];
  const rescueLeft = 190;
  const rescueWidth = 340;
  const gainLeft = 810;
  const gainWidth = 300;
  const bars = rows
    .map((row, index) => {
      const y = centers[index] - 17;
      const rescueBar = (row.rescue / 0.8) * rescueWidth;
      const gainBar = (row.gain / 0.08) * gainWidth;
      return `<text x="165" y="${centers[index] + 6}" class="label" text-anchor="end">${row.label}</text>
        <rect x="${rescueLeft}" y="${y}" width="${rescueBar.toFixed(1)}" height="34" rx="4" fill="${row.fill}" stroke="${row.stroke}" stroke-width="1.5"/>
        <text x="${(rescueLeft + rescueBar + 10).toFixed(1)}" y="${centers[index] + 6}" class="value">${pct(row.rescue)}</text>
        <text x="785" y="${centers[index] + 6}" class="label" text-anchor="end">${row.label}</text>
        <rect x="${gainLeft}" y="${y}" width="${gainBar.toFixed(1)}" height="34" rx="4" fill="${row.fill}" stroke="${row.stroke}" stroke-width="1.5"/>
        <text x="${(gainLeft + gainBar + 10).toFixed(1)}" y="${centers[index] + 6}" class="value">+${(row.gain * 100).toFixed(1)} pp</text>`;
    })
    .join("");
  const rescueTicks = [0, 0.4, 0.8]
    .map((value) => {
      const x = rescueLeft + (value / 0.8) * rescueWidth;
      return `<line x1="${x}" y1="190" x2="${x}" y2="455" class="grid"/><text x="${x}" y="180" class="tick" text-anchor="middle">${Math.round(value * 100)}%</text>`;
    })
    .join("");
  const gainTicks = [0, 0.04, 0.08]
    .map((value) => {
      const x = gainLeft + (value / 0.08) * gainWidth;
      return `<line x1="${x}" y1="190" x2="${x}" y2="455" class="grid"/><text x="${x}" y="180" class="tick" text-anchor="middle">${Math.round(value * 100)} pp</text>`;
    })
    .join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 600" role="img" aria-labelledby="title desc">
  <title id="title">Immediate repair rescue and held-out training gain are different estimands</title>
  <desc id="desc">CVoR produces a 79.1 percent independent rescue rate, far above Earliest and Random. After supervised training, however, CVoR gains 4.1 percentage points over Base, below Earliest at 5.5 and Random at 5.8.</desc>
  <rect width="1200" height="600" fill="#ffffff"/>
  <style>
    text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}
    .title{font-size:27px;font-weight:700}.subtitle{font-size:16px;fill:#526071}.panel{font-size:18px;font-weight:700}
    .label{font-size:16px;font-weight:600}.value{font-size:15px;font-weight:700;font-variant-numeric:tabular-nums}
    .tick{font-size:13px;fill:#64748b}.grid{stroke:#e2e8f0;stroke-width:1}.note{font-size:14px;fill:#64748b}
  </style>
  <text x="60" y="52" class="title">Immediate rescue and held-out training gain are different estimands</text>
  <text x="60" y="84" class="subtitle">Independent branch audit on the left; downstream ID gain versus Base on the right</text>
  <text x="190" y="135" class="panel">One-action repair rescue</text>
  <text x="810" y="135" class="panel">Held-out gain after SFT</text>
  ${rescueTicks}
  ${gainTicks}
  ${bars}
  <text x="60" y="520" class="note">CVoR’s selection-to-audit Spearman correlation is 0.734 across the full candidate pool.</text>
  <text x="60" y="548" class="note">The two panels intentionally use separate axes: branch-level rescue is not the same quantity as checkpoint-level learning utility.</text>
</svg>`;
}

fs.mkdirSync(imageDir, { recursive: true });
fs.mkdirSync(dataDir, { recursive: true });
fs.writeFileSync(
  path.join(dataDir, "confirmatory-v1.json"),
  `${JSON.stringify(publicData, null, 2)}\n`,
);
fs.writeFileSync(
  path.join(imageDir, "held-out-success.svg"),
  `${outcomesSvg()}\n`,
);
fs.writeFileSync(
  path.join(imageDir, "mechanism-vs-training.svg"),
  `${mechanismSvg()}\n`,
);

process.stdout.write(
  `${JSON.stringify({
    data: path.relative(root, path.join(dataDir, "confirmatory-v1.json")),
    images: [
      path.relative(root, path.join(imageDir, "held-out-success.svg")),
      path.relative(root, path.join(imageDir, "mechanism-vs-training.svg")),
    ],
  })}\n`,
);
