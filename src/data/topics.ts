export interface Topic {
  slug: string;
  title: string;
  description: string;
  tags: string[];
  focus: string[];
}

export const TOPICS: Topic[] = [
  {
    slug: "agents",
    title: "LLM Agents",
    description:
      "Tool use, agent runtime design, evaluation, context, and production patterns for systems that act across tools and environments.",
    tags: ["Agents"],
    focus: [
      "tool selection and orchestration",
      "agent evaluation and trajectory design",
      "context reuse and experience-augmented systems",
      "secure agent deployment",
    ],
  },
  {
    slug: "evaluation",
    title: "Evaluation",
    description:
      "Practical approaches to measuring model and agent capability with deterministic checks, rubrics, trajectories, and verifiable outcomes.",
    tags: ["Evaluation", "Post-Training", "RLHF"],
    focus: [
      "agent benchmark difficulty",
      "instruction-following checks",
      "reward and verification design",
      "evaluation data quality",
    ],
  },
  {
    slug: "post-training",
    title: "Post-Training",
    description:
      "SFT, RLHF, preference optimization, instruction following, reasoning traces, and data pipelines for shaping model behavior after pretraining.",
    tags: ["Post-Training", "RLHF", "Reinforcement Learning"],
    focus: [
      "RLHF objectives and engineering",
      "instruction-following data",
      "reasoning and thinking models",
      "unverifiable reward problems",
    ],
  },
  {
    slug: "rlhf",
    title: "RLHF and Preference Optimization",
    description:
      "Engineering notes and research synthesis on PPO, DPO, GRPO, reward modeling, preference data, and model behavior optimization.",
    tags: ["RLHF", "Reinforcement Learning"],
    focus: [
      "PPO, DPO, and GRPO",
      "preference data pipelines",
      "reward modeling",
      "creative and unverifiable rewards",
    ],
  },
  {
    slug: "generative-ui",
    title: "Generative UI",
    description:
      "How AI systems can produce, steer, and execute user interfaces with structured representations and practical product constraints.",
    tags: ["UI", "Generative", "UX", "Product"],
    focus: [
      "structured UI representation",
      "action execution",
      "steering versus generation",
      "LLM product interfaces",
    ],
  },
];
