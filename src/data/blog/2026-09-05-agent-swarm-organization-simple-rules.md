---
author: Jing Lu
pubDatetime: 2026-09-05T12:00:00-07:00
title: "What Makes an Agent Swarm Work? Simple Rules, Information Boundaries, and the Value of Verification"
featured: true
draft: false
tags:
  - AI
  - LLM
  - Agents
  - Evaluation
  - Reinforcement Learning
  - ML Engineering
description: "A theory-and-experiment study of agent swarm organization: operational roles, marginal-contribution incentives, selective information sharing, and why verification remains the limiting assumption."
---

<style>
  #article table {
    display: block;
    max-width: 100%;
    overflow-x: auto;
  }
  #article .katex-display {
    overflow-x: auto;
    overflow-y: hidden;
    padding-block: 0.3rem;
  }
</style>

An agent swarm can produce a surprising amount of activity without making
much progress. Four agents search, write reports, exchange opinions, and
eventually agree. The shared folder fills up. The final answer may still rely
on a missing relationship, an incompatible date, or one unsupported guess
that traveled through several agents.

Adding a planner, researcher, critic, and editor gives that activity familiar
names. It does not explain why their individual actions should improve the
result. I wanted a more concrete account of organization:

> Can a very simple rule for each agent, combined with a very simple rule for
> the group, make useful specialization emerge? What information must they
> share for that to work?

This led to a literature review, several mathematical models and exact
checks, and a sequence of small language-model experiments. The most useful
result is a conditional one. **Simple local incentives can align individual
improvement with team progress when contribution is well defined and
observable. In open-ended tasks, reliably measuring that contribution becomes
the difficult part.**

The experiments did not identify a universal winning organization. Sharing
raw evidence solved one additional question in an eight-question study, at
higher token cost. In a later comparison with nearly equal measured cost,
prioritizing verification produced one additional correct answer and one
additional wrong submission. Those outcomes are worth understanding precisely
because they resist a clean architecture success story.

I will separate three kinds of evidence throughout: established theory,
exact checks in deliberately small mathematical worlds, and exploratory LLM
measurements. None can substitute for the others.

## Table of contents

## 1. Define progress before designing the organization

The recurring swarm problem is straightforward to state:

> Given what the team has already learned, how should its next action add
> accepted progress toward the task, within the remaining budget?

This formulation makes several popular proxies look inadequate. A new
document can repeat an old fact. A new fact can be irrelevant. A correct
candidate can lack the evidence needed to justify it. A unanimous panel can
share one mistaken premise.

For a retrieval task, it helps to distinguish the following objects:

| Object             | Example                                                           | What it establishes              |
| ------------------ | ----------------------------------------------------------------- | -------------------------------- |
| Observation        | A versioned passage returned by a search or read                  | What the tool supplied           |
| Hypothesis         | This question may refer to a particular television adaptation     | A candidate route to investigate |
| Supported relation | A passage connects an actor to a role in the specified adaptation | One link in the argument         |
| Completed task     | The answer satisfies the question's required constraints          | The actual objective             |

The distinction is operational. An observation belongs in an evidence store.
A hypothesis belongs in a candidate set. A supported relation needs its
premises and scope. A completed task needs an acceptance rule. Putting all
four into one undifferentiated conversation makes it easy for status to change
without any new evidence.

Even the word _better_ needs a definition. A system that answers more
questions while also making more unsupported claims may or may not be an
improvement. Accuracy, abstention, error cost, latency, and total inference
cost should be specified before selecting an architecture.

## 2. Useful roles describe operations and information

There is a substantial literature on roles that has little to do with
imitating human professions.

[ROMA](https://proceedings.mlr.press/v119/wang20f.html) learns stochastic role
representations that condition individual policies. Its regularizers
encourage identifiable specialization in a multi-agent reinforcement-learning
environment. [RODE](https://arxiv.org/html/2010.01523) defines roles through
restricted action spaces, learning action representations from their effects
on the environment and other agents. A higher-level selector changes roles
on a slower time scale.

The important transfer is the definition of a role: **a constraint on what an
agent observes, can do, or is responsible for producing**. The empirical
results of those papers involve trained policies, including StarCraft
micromanagement tasks. They do not establish that changing the persona prompt
of a frozen LLM produces the same kind of learned specialization.

[GPTSwarm](https://proceedings.mlr.press/v235/zhuge24a.html) offers a useful
LLM-specific representation. Nodes are operations, including model calls and
other processing functions; edges carry information. Prompts and connections
can be optimized. A node in such a graph does not need to be a persistent
autonomous individual.

For a research system, I would start with a small operation vocabulary:

| Operation | Input                                       | Output                                         | Useful restriction                              |
| --------- | ------------------------------------------- | ---------------------------------------------- | ----------------------------------------------- |
| Acquire   | An unresolved information need              | New tool observations                          | Pay the actual search or execution cost         |
| Connect   | Existing observations                       | A candidate relation and its premises          | Keep inference distinguishable from source text |
| Challenge | A candidate and its dependencies            | A contradiction, failed test, or remaining gap | Target a specific claim                         |
| Commit    | A proposed result and evidence              | A versioned accepted item                      | Apply the task's acceptance checks              |
| Route     | Current dependencies and available capacity | The next work allocation                       | Account for conflicts and remaining budget      |

These are five operations, not a prescription to launch five agents. The same
worker can acquire in one step and challenge in the next. A deterministic
function may perform part of commit. An expensive semantic check may need a
model. Specialization can follow tools, data access, and observed competence
without turning every operation into a job title.

### There are several different meanings of “the swarm evolves”

An organization can adapt during a task by changing assignments. A search
procedure can improve an entire workflow across development tasks. A training
algorithm can update worker or orchestrator policies. These are different
optimization problems.

[AFlow](https://arxiv.org/html/2410.10762) searches code-represented workflows
using execution feedback. [G-Designer](https://proceedings.mlr.press/v267/zhang25cu.html)
generates task-conditioned communication topologies. These approaches make
organization an optimization variable, but they do not by themselves prove
that identical workers following a local rule will self-organize optimally
during a single task. Their search and training costs also belong in a fair
comparison.

The experiments in this post use frozen model calls with explicit information
and action rules. They do not train emergent roles or evolve model weights.

## 3. What related research says about communication

The literature is more useful as a map of mechanisms than as a ranking of
organizational charts.

| Research direction                           | Representative work                                                                                                                               | What it helps answer                                                         |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Communication as a cost                      | [AgentPrune](https://arxiv.org/html/2410.02506), [sparse debate topologies](https://aclanthology.org/2024.findings-emnlp.427/)                    | Which messages and rounds can be removed in a tested protocol?               |
| Information distributed across agents        | [SILO-BENCH](https://aclanthology.org/2026.acl-long.1354/)                                                                                        | Can agents coordinate when relevant knowledge is separated?                  |
| Architecture interacting with task structure | [Towards a Science of Scaling Agent Systems](https://arxiv.org/html/2512.08296v2)                                                                 | How do coordination patterns behave on different task families?              |
| Shared work state                            | [Contract Net](https://doi.org/10.1109/TC.1980.1675516), [blackboard systems](https://www.sciencedirect.com/science/article/pii/0004370285900633) | How are tasks advertised, selected, and integrated?                          |
| Replies with restricted visibility           | [LLM-based Multi-Agent Blackboard System](https://arxiv.org/html/2510.01285v2)                                                                    | Can requests be public while candidate responses remain separated?           |
| Debate versus independent aggregation        | [Should we be going MAD?](https://proceedings.mlr.press/v235/smit24a.html)                                                                        | Does interaction add value beyond strong sampling and aggregation baselines? |

These papers change different things. Pruning messages is not equivalent to
discovering semantic redundancy. A task-conditioned graph is not necessarily
reorganized during execution. A successful blackboard system does not isolate
the causal effect of hiding replies. A topology comparison can also change
the controller, number of calls, stopping rule, or aggregation procedure.

The human collective-intelligence literature supplies an especially useful
distinction. [Facts and Figuring](https://pubsonline.informs.org/doi/10.1287/orsc.2015.0980)
separates exploration of an information space from exploration of possible
explanations. Its network experiment found that clustering could help the
former while inhibiting the latter. [Intermittent interaction experiments](https://pmc.ncbi.nlm.nih.gov/articles/PMC6126746/)
also show that the effects of communication depend on the interaction
schedule and access to a participant's own best solutions.

These are human experiments. They motivate separating facts, interpretations,
and memory in an LLM study; they do not prove a particular masking schedule
for language agents.

Engineering accounts point to similar concrete issues.
[Anthropic's research-system report](https://www.anthropic.com/engineering/multi-agent-research-system)
describes duplicate work and omissions when subtask boundaries are vague.
Its [parallel compiler-building account](https://www.anthropic.com/engineering/building-c-compiler)
describes coordination through task locks, alongside a case where a common
blocker limited useful parallelism. These are informative implementation
reports, not controlled proofs that locks or a lead-agent hierarchy are
generally optimal.

The resulting research question is narrower and testable: **which information
should change the next action, and what does sharing it cost?**

## 4. Full information cannot hurt an unconstrained optimum

Before arguing for information barriers, we should rule out an overly strong
claim.

Suppose an agent sees history $h$ in a restricted system and richer history
$H$ in a fully informed system. Assume that $h=g(H)$, extra information is
free, and the agent is allowed to ignore it. For every restricted policy
$\pi(h)$, the richer system can implement

$$
\pi'(H)=\pi(g(H)).
$$

It can reproduce the restricted system exactly. The same construction applies
to multiple agents and, by induction, to multiple rounds. Therefore,

$$
\sup_{\pi\in\Pi_{\mathrm{full}}}\mathbb E[F(\pi)]
\geq
\sup_{\pi\in\Pi_{\mathrm{restricted}}}\mathbb E[F(\pi)].
$$

This is a policy-class containment argument. It concerns the best feasible
policy, not the behavior of a particular prompted model.

An actual LLM pays to read context. It may fail to ignore misleading material,
adopt a peer's framing, or spend its remaining budget reconciling redundant
reports. Restricting its input can improve that bounded implementation. Such
a result would demonstrate an interaction between information and the
decision procedure; it would not show that information has intrinsically
negative value.

There is an equally simple warning in the other direction. If the task is to
output the XOR of two independent fair bits, seeing both permits perfect
accuracy. Permanently hiding one limits any policy to one-half accuracy.
Preventing conformity by withholding necessary evidence can destroy
solvability.

The practical target is **sufficient decision-relevant information at an
acceptable cost**. For some operations that may be a count or an identifier.
For others it may be an entire dependency chain.

## 5. A minimal mechanism with an actual guarantee

Consider a deliberately restricted task family. There is a finite collection
of work items $r$, each with known nonnegative value $v_r$. Agent $i$ chooses
one feasible action $a_i$ from its finite action menu; that action covers a
set of items. The team's value is the value of the union:

$$
F(a)=\sum_{r\in\bigcup_i a_i}v_r.
$$

Completing the same item twice does not double its value. The menus may differ
across agents, representing different tools or capabilities. The model assumes
we know which items an action covers; this is a strong assumption that we
will later remove.

Give each agent its marginal contribution:

$$
u_i(a)=F(a)-F(a_{-i}),
$$

where $a_{-i}$ is the configuration with agent $i$ removed. In a coverage
task, this is the total value of items currently covered only by that agent.

The rules are small:

1. **Individual:** change action only when doing so strictly increases your
   marginal contribution.
2. **Group:** commit one such change at a time, using the current state, and
   eventually give every agent with an improvement the opportunity to act.

There is no requirement to invent different personalities. The system must
implement the payoff correctly, and agents must respond to it. This is
distributed utility design. It does not establish truthful reporting of
private information, resistance to collusion, or compliance by an arbitrary
LLM given a sentence about rewards.

### Individual improvement equals team improvement

Hold the other agents fixed and let agent $i$ switch from $a_i$ to $b_i$.
Writing its utility change as $\Delta u_i$,

$$
\begin{aligned}
\Delta u_i
&=[F(b_i,a_{-i})-F(a_{-i})]\\
&\quad-[F(a_i,a_{-i})-F(a_{-i})]\\
&=F(b_i,a_{-i})-F(a_i,a_{-i}).
\end{aligned}
$$

The counterfactual baseline cancels. The team's value is an **exact
potential**: every unilateral utility improvement increases it by the same
amount. This identity works for any $F$, not just coverage.

With finitely many configurations, strict increases cannot revisit a previous
configuration. An improvement path therefore terminates. If profitable moves
are reliably found and scheduled, the endpoint is a pure Nash equilibrium:
no agent can improve by switching alone.

Termination is not an efficiency guarantee. The number of configurations can
grow exponentially, and discovering a better feasible action can itself be
expensive. The theorem has not paid for LLM deliberation, search, or checking.

### Coverage gives a one-half equilibrium bound

Coverage has diminishing returns: adding a fixed action contributes no more
when more items are already covered. This is submodularity.

Let $A$ be an equilibrium and $O$ an optimal feasible configuration. We may
evaluate unions of agent-labeled actions as a mathematical intermediate,
even though an agent cannot execute two alternative actions simultaneously.
Monotonicity and diminishing returns give

$$
\begin{aligned}
F(O)
&\leq F(A\cup O)\\
&\leq F(A)+\sum_i\left[F(A\cup\{o_i\})-F(A)\right]\\
&\leq F(A)+\sum_i u_i(o_i,A_{-i}).
\end{aligned}
$$

At equilibrium, $u_i(o_i,A_{-i})\leq u_i(A)$. Moreover,
$\sum_i u_i(A)\leq F(A)$: exclusively covered items are counted once, and
multiply covered items contribute nothing to that sum. Consequently,

$$
\boxed{F(A)\geq \frac{\mathrm{OPT}}{2}.}
$$

This is a coverage specialization of the classical valid-utility framework
associated with [Vetta](https://web.mit.edu/6.454/www/www_fall_2004/gametheory/VettaNashEquilibria.pdf).
It is not a new universal theorem about LLM swarms. In particular, one-half
is a ratio of coverage values within the same action space, not a promise of
50% question-answering accuracy or an advantage over a single agent.

The bound can be attained. Let two items $x,y$ each have value one. Agent 1
can choose nothing or $x$; agent 2 can choose $x$ or $y$. At
$(\varnothing,x)$, agent 1 gains nothing by duplicating $x$, and agent 2
gains nothing by moving to $y$. The configuration is an equilibrium of value
one, although $(x,y)$ has value two. Moving between them requires an
indifferent intermediate step or coordination.

The research checks enumerated 14,400 small coverage games, 115,200 unilateral
changes, and 31,115 equilibria without violating the identities or bound.
These are exact finite checks alongside a mathematical proof, not a proof
assistant formalization and not LLM trials.

## 6. Better equilibrium guarantees can sacrifice steady progress

Marginal contribution is not the only simple local utility. Let $n_r$ be the
number of agents covering item $r$, and use a common reward table $f$:

$$
U_i(a)=\sum_{r\in a_i}v_r f(n_r).
$$

Marginal contribution uses $f(1)=1$ and $f(k)=0$ for $k\geq2$. Equal sharing
uses $f(k)=1/k$. An optimized table can improve the worst equilibrium within
this coverage-game family. For four agents,

$$
(f(1),f(2),f(3),f(4))=
\left(1,\frac{13}{31},\frac{8}{31},\frac{6}{31}\right)
$$

gives a worst-equilibrium ratio of $31/49\approx63.27\%$.
The utility design and its guarantees are established in
[Ramaswamy, Paccagnan, and Marden](https://arxiv.org/pdf/1710.01409v3).
The research independently checked rational certificates and tight examples
for three reward rules at each agent count from two through six.

| Agents | Marginal contribution | Equal sharing | Optimized table |
| -----: | --------------------: | ------------: | --------------: |
|      2 |                   1/2 |           2/3 |             2/3 |
|      3 |                   1/2 |           3/5 |            7/11 |
|      4 |                   1/2 |           4/7 |           31/49 |
|      5 |                   1/2 |           5/9 |           55/87 |
|      6 |                   1/2 |          6/11 |       1031/1631 |

The table describes worst-case equilibrium coverage, not observed language
model success. The local rule needs only item values and accurate occupancy
counts. Its potential is

$$
\Phi(a)=\sum_r v_r\sum_{k=1}^{n_r(a)}f(k).
$$

Unilateral improvement increases $\Phi$, but $\Phi$ need not equal $F$.
For a concrete example, let $x$ have value one and $y$ value three. Agent 2
always covers $y$; agent 1 chooses $x$ or $y$. Under equal sharing, moving
from $x$ to $y$ raises agent 1's utility from one to $3/2$, while the team's
coverage falls from four to three.

This distinction matters if the system has a deadline. Improving the worst
eventual equilibrium and making every intermediate state better are different
design objectives. Neither automatically maximizes useful work within eight
tool calls.

## 7. Information barriers can break the mechanism

The potential argument relies on correct contribution estimates. Local
visibility can invalidate them even when every agent follows the rule
perfectly.

Take three agents, each choosing item $x$ or $y$. Agent 0 sees only agent 1,
agent 1 sees only agent 2, and agent 2 sees only agent 0. Each receives a
perceived reward of one when its choice differs from the agent it observes,
and zero otherwise.

Each wants to choose differently from its observed neighbor. Three such
requirements cannot all hold on an odd cycle with two choices. There is no
pure equilibrium. Worse, strictly improving _asynchronous_ changes can cycle:

```text
(y,x,x) → (y,y,x) → (x,y,x) → (x,y,y)
        → (x,x,y) → (y,x,y) → (y,x,x)
```

Each arrow changes one agent's action and raises that agent's perceived
utility from zero to one. The perceived contributions no longer correspond
to a shared team potential.

The finite visibility study checked 64 directed graphs across 216 action-menu
combinations, or 13,824 games. The two directed three-cycles supplied the
no-equilibrium examples in that family. This illustrates the distinction
between a bound on equilibria and a guarantee that suitable dynamics reach
one, also emphasized in
[utility design with arbitrary information networks](https://arxiv.org/html/2501.17385v1).

### Share the sufficient state for the decision

In the coverage model, an agent does not need all its peers' reasoning. It
needs the accurate occupancy of the items relevant to its alternatives,
with its own occupancy handled correctly. Those counts are sufficient to
recover the same marginal contribution as full action visibility.

This suggests a narrower public interface than broadcasting every report:

```text
item identifier
current occupancy or reservation
accepted result and version
dependencies or conflicts relevant to the next action
```

The counts must describe a consistent current snapshot. A worker can compute
a promising change against one version and discover at commit time that a
peer has already taken it. Conflict checks are part of the mechanism.

Even full visibility does not rescue uncoordinated simultaneous changes. If
two agents both cover $x$, both may see a gain from switching to empty $y$.
If both switch together, neither gets unique coverage. They can oscillate
between the two items. Serializing conflicting commits, or using a valid
reservation protocol, addresses a different problem from merging messages.

Real work also has semantic overlap. Two differently worded subtasks may
cover the same need; two identically named tasks may concern different
versions. An occupancy counter cannot solve that interpretation problem.

## 8. Preserve provenance without mistaking it for truth

A separate simple rule helps with repeated evidence. Give each original
observation a stable identifier and version. A derived message records the
set of original observations on which it depends:

$$
L(m)=\bigcup_{p\in\mathrm{parents}(m)}L(p).
$$

The receiving evidence store merges these sets by union. Because union is
idempotent, commutative, and associative, retransmission, order changes, and
multiple paraphrase paths do not create extra original observations. This is
the basic algebra behind a grow-only set in the
[CRDT framework](https://www.lip6.fr/Marc.Shapiro/papers/RR-7687.pdf).

The rule solves provenance duplication. It does not make distinct sources
independent or true. Two websites can repeat the same original reporting.
Two separate test executions can legitimately produce identical text.
Deduplicating by message wording or domain misses the causal distinction.
Related repeated-information effects predate LLMs in the
[data-incest literature](https://arxiv.org/abs/1309.6687v3).

A small probability calculation makes the issue visible. Suppose three
conditionally independent binary observations are each correct with
probability $p>1/2$, with a balanced prior. If the first observation appears
three times in a five-message conversation, message-majority always follows
that first observation and has accuracy $p$. Counting the three unique
observations gives MAP accuracy

$$
3p^2(1-p)+p^3=3p^2-2p^3.
$$

At $p=0.75$, the two accuracies are 75% and 84.375%. This is a comparison of
specified mathematical decision rules. It does not predict that an LLM will
naively count the duplicated message as three independent votes.

Indeed, a paired 48-call representation microbenchmark did **not** show that
accuracy benefit. The model made the correct MAP decision on 24/24 raw-message
cases and 23/24 deduplicated-board cases. The board used 20.99% fewer recorded
tokens. That supports a cost observation for those inputs, not an accuracy
improvement and not a general claim that summaries preserve everything.

Derived reasoning can still be useful. Even if it adds no new information to
an ideal reasoner, it may save a bounded model substantial computation. Keep
the reasoning available when helpful; keep its evidential lineage explicit.

## 9. The real experiment: change visibility, then execute new actions

To move beyond toy models, I used questions from
[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) with its frozen
100,195-document corpus. The local retrieval implementation used SQLite FTS5
BM25, returning up to four documents and a query-dependent passage of up to
2,200 characters per document. This differs from the official retrieval and
grading stack, so the absolute scores are not an official benchmark
reproduction.

The model request was `gpt-5.4-mini` with low reasoning effort. There was no
immutable model snapshot or controllable provider seed. Each complete policy
had an admission cap of 192,000 input-plus-output tokens and eight tool
requests. Equal caps did not force equal actual usage.

Eight question IDs were fixed before collection and had not been used in the
earlier pilot. This was an ID-separated development set, not a topic-held-out
sample or evidence of absence from model pretraining. Each condition had one
trajectory per question.

### Four conditions, with a shared starting point

For the three swarm conditions, one neutral decomposition generated four
initial searches. Each worker then wrote a private report about its own
observations. That common prefix was frozen before branching.

| Condition                  | What a worker saw before its next action                             |
| -------------------------- | -------------------------------------------------------------------- |
| P: private exploration     | Common plan, its own passages, its own report                        |
| E: shared evidence         | The same material plus all initial raw passages and executed queries |
| H: shared reports          | The same raw evidence as E, plus peers' candidate reports            |
| S: sequential single agent | Its own adaptively acquired observations and action history          |

Each P/E/H worker chose and executed a new action under its assigned
visibility. I did not replay the old queries and pretend they represented a
new policy. All workers used the same operational instruction: choose the
next action most useful for resolving the root question's outstanding
constraints.

The final synthesizer received all raw observations actually acquired by its
condition, with the same prompt and no arm label or peer reports. Thus P was
private during exploration, but still aggregated the team's evidence at the
end. It was not a permanently disconnected system.

![Visibility experiment: a shared prefix branches into private evidence, shared evidence, and shared reports before new searches and raw-evidence synthesis.](/images/agent-swarm-organization/visibility-protocol.svg)

### The observed gain was one question, with higher cost

| Condition              | Correct | Wrong submission | UNKNOWN | Mean complete-policy tokens |
| ---------------------- | ------: | ---------------: | ------: | --------------------------: |
| P: private exploration |     1/8 |              1/8 |     6/8 |                      67,458 |
| E: shared evidence     |     2/8 |              0/8 |     6/8 |                      96,118 |
| H: shared reports      |     2/8 |              1/8 |     5/8 |                      97,555 |
| S: sequential agent    |     1/8 |              2/8 |     5/8 |                      92,934 |

E's additional success came from one question. Its average recorded token
usage was about 42.5% higher than P's. H did not solve an additional question
beyond E. These counts do not establish a population-level benefit,
noninferiority, or an optimal amount of sharing.

P/E/H used eight tool requests each; S averaged 6.5. Two S intermediate calls
attempted an unavailable user-input tool and were retained as failures, with
final synthesis continuing on available evidence. Consequently, this is also
not a comparison against every possible strong sequential policy.

All correct final answers matched the reference after conservative
normalization. Other substantive candidates were checked by two judges blind
to the condition; they agreed. That agreement concerns equivalence to the
benchmark short answer, not independent verification of every real-world
claim in the reasoning.

### A correct name can conceal a missing evidential link

The additional successful question involved identifying an actor through a
television-adaptation chain. E acquired a passage explicitly connecting
Benjamin Whitrow to Mr. Bennet in the 1995 adaptation of _Pride and Prejudice_.
Its final answer had the relevant actor–role evidence.

H returned the same correct name, but the text and URLs supplied to its
final synthesizer did not contain that name. Its citations did not establish
the required actor–role relation. The output alone cannot tell us exactly
whether the name came from parametric knowledge or some other internal
inference.

I then repeated the final synthesis three times for each condition using
byte-identical original inputs. E returned the correct name 3/3 times. H did
so 2/3 times and once returned an actor from a different adaptation. P and S
were each 0/3. This was a post hoc check of the one selected gain case; it
measures neither fresh search-policy variability nor a general success rate.

The lesson is a measurement distinction:

$$
\begin{gathered}
\text{correct benchmark answer}\\
\neq\ \text{supported relation}\\
\neq\ \text{fully verified constraint chain}.
\end{gathered}
$$

The broader separation between answer correctness and citation support also
appears in [ALCE](https://aclanthology.org/2023.emnlp-main.398/). A system can
succeed on one dimension and fail on another.

## 10. Make verification consume the same budget

The visibility result suggested a practical intervention: once a plausible
candidate exists, spend the next action on its most important missing
relationship. But supplying an extra verification step after the baseline's
budget is exhausted would give the intervention a resource advantage.

I therefore ran a final-action comparison on the same eight development
questions. For each question, I retained seven tool observations from the H
trajectory and removed its remaining action, result, and original final
answer. The retained second-wave actions had all originally depended only on
the common initial evidence, so removing the fourth worker did not invalidate
their histories.

A new shared diagnosis proposed a candidate and a critical missing relation.
Both policies received exactly that diagnosis and the same raw prefix:

| Policy                   | Final-action instruction                                                                                                      |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| B: free choice           | Choose the most useful next action; verification, another search route, reading, and stopping are allowed                     |
| V: verification priority | Prefer establishing or refuting the candidate's critical missing relation; fall back when no suitable candidate or gap exists |

B was allowed to verify. V did not receive privileged evidence. Each policy
was sampled twice per question, its chosen tool action was actually
executed, and a fresh final synthesizer saw only its raw evidence.

The full accounting included the old prefix, shared diagnosis, action
selection, and final synthesis. The cap remained eight tools and 192,000
tokens. An additional N condition synthesized directly from the seven-tool
prefix once per question. N is a useful ablation, but uses less actual
resource and lacks the diagnosis cost.

### One more correct answer, one more wrong answer

| Outcome                        | B: free choice | V: verification priority | N: seven-tool synthesis |
| ------------------------------ | -------------: | -----------------------: | ----------------------: |
| Trajectories                   |             16 |                       16 |                       8 |
| Correct                        |              3 |                        4 |                       1 |
| Wrong submission               |              0 |                        1 |                       0 |
| UNKNOWN                        |             13 |                       11 |                       7 |
| Mean complete-policy tokens    |     125,644.75 |               125,578.19 |               82,823.75 |
| Mean tool requests             |         7.9375 |                   7.9375 |                       7 |
| Intermediate protocol failures |              1 |                        1 |                       0 |

![Final-action outcomes: free choice has 3 correct, 0 wrong, and 13 abstentions; verification priority has 4 correct, 1 wrong, and 11 abstentions.](/images/agent-swarm-organization/last-action-outcomes.svg)

B and V differed in mean complete-policy token usage by only about 0.053%.
That is observed cost similarity, not exact server-side token matching.
Each suffered one unavailable-tool failure; both continued to final synthesis,
and neither failure occurred on the question producing the correctness gain
or the wrong submission.

Paired by question and replicate index, three pairs were jointly correct,
one favored V, and twelve were jointly unsuccessful. The replicate index
does not couple provider randomness. There are eight questions with repeated
calls, not sixteen independent tasks. All paired final actions and evidence
sets differed, so the experiment compares sampled prompted policies; it does
not attribute every wording difference to the instruction alone.

### The ranking depends on the cost of a wrong submission

As a post hoc sensitivity analysis, assign one point for a correct answer,
zero for UNKNOWN, and $-\lambda$ for a wrong submission. Ignoring the tiny
observed token-cost difference,

$$
\begin{aligned}
\widehat U_B&=\frac3{16},\\
\widehat U_V&=\frac{4-\lambda}{16},\\
\widehat U_V-\widehat U_B&=\frac{1-\lambda}{16}.
\end{aligned}
$$

V scores higher if wrong answers carry no penalty, ties at $\lambda=1$, and
scores lower at $\lambda=2$. This is not a new primary metric chosen to
declare a winner. It shows why “a better group state” cannot remain undefined.

### What the final actions actually revealed

**Targeted relation retrieval sometimes worked.** On the actor question,
one B branch and both V branches searched the specific adaptation's cast and
retrieved the explicit missing actor–role relation. All three answered
correctly. The other B branch pursued a different birth-year interpretation
and abstained. These are three successes on one shared case, not three
independent demonstrations. The cast passage also did not automatically
validate every other clue in the question.

**Finding a document was not enough.** On another question, all four B/V
searches found the relevant document, but the query-dependent extractor
returned a bibliography-like passage rather than the needed coach–year
relationship. A targeted read had recovered that relationship in an earlier
diagnostic. In this experiment the document first appeared on the final tool
call, so that additional read would exceed the budget. A different search
query might have exposed the right passage immediately; the result does not
prove that every one-call strategy must fail.

**Verification instructions did not remove contradictory framing.** One
question required a university founded in the seventeenth century. The
shared evidence placed the candidate university's founding in 1477, yet
several last actions continued searching around that candidate. The conflict
was visible without consulting the reference answer.

**A shared diagnosis could still supply a bad route.** For a foundation
question, all four searches pursued the same incorrect organization family.
That is observed convergence on an unproductive hypothesis. Because the
diagnosis itself was not experimentally varied, the data do not separately
identify its causal anchoring effect.

**Citation integrity remained imperfect.** V's one wrong submission cited
documents about a different entity. Another V abstention included a
nonexistent document identifier. A verification-priority prompt did not
enforce a valid citation contract.

All 30 successful final actions selected search. None selected read or
finish. The experiment therefore provides evidence about these search
decisions, not a comprehensive evaluation of reading and stopping policies.

## 11. Why a one-step contribution rule can fail even with perfect inference

One tempting response is to improve the model's estimate of which query will
help most. Better estimates matter, but a second problem remains: some pieces
of evidence are valuable only together.

The simplest coordination counterexample has two agents choosing zero or
one. The team receives one only if both choose one. From $(0,0)$, neither
unilateral change has positive value. Marginal-contribution improvement can
stop at zero even though the optimum is one. The potential identity still
holds; the coverage-quality bound does not, because this objective has
complementarity rather than diminishing returns.

Here is an information-acquisition version that makes the loss arbitrarily
large, despite a known prior and exact Bayesian inference.

Let the answer $Y$ be uniform over $m$ possibilities. Draw an independent
uniform $X$, and define

$$
Z=(Y-X)\bmod m.
$$

There are three equally priced queries, with budget for two:

| Query | Returned information                                           |
| ----- | -------------------------------------------------------------- |
| $q_X$ | The value of $X$                                               |
| $q_Z$ | The value of $Z$                                               |
| $q_L$ | The answer $Y$ with probability $\delta$, otherwise an erasure |

The hint's reveal event is independent of $X$ and $Y$. On its own, either
share leaves the answer uniform, so immediate optimal answer accuracy remains
$1/m$. The hint raises expected accuracy to

$$
\delta+(1-\delta)/m.
$$

A policy that greedily maximizes the next step's expected answer accuracy
therefore picks the hint. If it is erased, only one query remains, and one
share still cannot reveal the answer. The greedy two-query value is

$$
V_{\mathrm{greedy}}=\delta+(1-\delta)/m.
$$

The optimal policy queries $X$ and $Z$, then computes
$Y=(X+Z)\bmod m$. Its accuracy is one. Taking $\delta=1/m^2$ makes the
greedy-to-optimal ratio tend to zero. At $m=20$,

$$
\begin{aligned}
V_{\mathrm{greedy}}&=419/8000=5.2375\%,\\
V_{\mathrm{optimal}}&=100\%.
\end{aligned}
$$

These are synthetic mathematical values, not LLM measurements. Twenty-four
finite instances were checked with rational arithmetic and exact adaptive
decision trees, using the most favorable tie-breaking for greedy.

The counterexample explains the structural issue. Before seeing $X$, learning
$Z$ has zero immediate decision value. After seeing $X$, it has value
$1-1/m$. Its marginal value **increases** with the evidence already acquired.

[Adaptive submodularity](https://arxiv.org/html/1003.3967v5) supplies useful
greedy approximation guarantees when conditional marginal returns diminish.
For the normalized, adaptive-monotone, adaptive-submodular cardinality-budget
setting, exact adaptive greedy obtains the familiar $1-1/e$ guarantee.
Those assumptions, including the probability model needed to evaluate
conditional gains, do not come free with a natural-language prompt. The
reference here is the corrected v5 text; minimum-cost coverage has additional
conditions and different bounds.

[Submodular Surrogates for Value of Information](https://ojs.aaai.org/index.php/AAAI/article/view/9694)
shows another principled route: construct a tractable surrogate for a
specified decision problem. Its known probabilistic model and decision
utility are substantive inputs, not a general-purpose verifier for open web
research.

This is why a good organization may need to reserve a bundle of dependent
actions. “Find the document, then inspect the relevant section” can be the
useful unit, even when the first action has little immediate answer value.

## 12. Verification is the limiting interface

We can state a clean invariant for a shared result ledger. Fix a consistent,
versioned evidence snapshot $E$. A proposed claim $c$ arrives with supporting
material $p$. Accept it only when a verifier $V(E,c,p)$ returns one.

If the verifier is sound,

$$
V(E,c,p)=1\ \Longrightarrow\ E\models c,
$$

and the initial accepted set contains only supported claims, then induction
shows that every later accepted claim is supported. The agent's persona is
irrelevant to this invariant.

The premise is doing the work. A substring check can establish that a quoted
span occurs in a source. It cannot establish that the span supports the
particular relation, that an omitted qualifier is harmless, or that the source
is correct. Source support is a claim relative to premises. World truth
requires something further.

A post hoc relation-control experiment tested ten focal passage cases twice
each. Controls included deleting the actor's name, retaining a name while
removing the needed relation, using the wrong adaptation, and substituting a
fictional actor within an otherwise explicit relation. The 20 support labels
matched the frozen case labels. All eight accepted cases supplied nonempty
exact quotes. Two rejected-case explanations stitched text with ellipses and
failed a continuous-substring check.

That is evidence of a local capability to distinguish these supported and
unsupported relations. It is not a general soundness certificate, an
independent test distribution, or a demonstration that the system can discover
all missing relationships end to end.

### More reviewers do not automatically give independent checks

If all reviewers fail on the same latent ambiguity with probability $p$,
requiring all of them to accept can still have error probability $p$, not
$p^k$. Fresh sessions remove shared conversation history; they do not
automatically remove common training data, priors, or interpretation errors.

A valid probabilistic ledger guarantee can instead assume an error bound at
each step conditional on the entire past:

$$
P(\text{false acceptance at }t\mid\mathcal H_{t-1})\leq\epsilon_t.
$$

Then the union bound gives

$$
P(\text{any false acceptance by }T)\leq\sum_{t=1}^{T}\epsilon_t.
$$

Independence is unnecessary for that argument. What is necessary is a valid
conditional error bound. A model's self-reported confidence does not establish
one. This connects directly to the distinction between verifier design and
verified verifier performance in the site's
[horizon-scaling study](/posts/2026-08-14-verifier-error-horizon-scaling/).

### Contribution estimates have the same hidden requirement

Suppose every relevant utility estimate satisfies
$|\widehat u_i-u_i|\leq\epsilon_i$ on the state being evaluated. An
estimated improvement exceeding $2\epsilon_i$ must be a true improvement,
because estimating the old and new actions introduces at most
$2\epsilon_i$ difference error.

If all candidate changes have been checked and none exceeds that threshold,
every true unilateral gain is at most $4\epsilon_i$. The coverage argument
then yields

$$
F(a)\geq\frac{\mathrm{OPT}-4\sum_i\epsilon_i}{2}.
$$

This is a useful robust statement. It also makes the missing premise explicit:
we need an error guarantee across the relevant alternatives and states. A
handful of correct critiques cannot provide that uniform guarantee.

### Why finite black-box tests cannot certify arbitrary future tasks

Let $D$ be the finite set of situations already validated. A situation
includes a task, its history, and an action whose contribution is being
estimated. Choose an unobserved future situation $x^*$.

Without restrictions on the task family, distribution, or contribution
function, two models can agree on every observation in $D$ while satisfying

$$
g_0(x^*)=0,\qquad g_1(x^*)=1.
$$

The same observations induce the same estimate $v$ at $x^*$. But

$$
1\leq |v-g_0(x^*)|+|v-g_1(x^*)|,
$$

so at least one compatible world has error at least one-half. More finite
tests enlarge $D$; absent further structure, another unobserved case remains.

This argument limits a distribution-free, uniform future guarantee. It does
not make the observed labels arbitrary, and it does not say that empirical
research must stop. Specifying a task distribution permits statistical
generalization claims. Restricting tasks to a formal environment can permit
sound checking. What fails is upgrading finite open-ended demonstrations into
a guarantee about every future semantic contribution.

The bottleneck is consequently more precise than “we need smarter agents.”
The mechanism needs a sufficiently reliable relationship between its local
score and actual task value, including the cost of learning that value.

## 13. What organization I would test next

The evidence suggests a concrete design hypothesis: a small public state
for coordination and accepted evidence, with exploration views selected by
the dependencies of each action. The worker rule can remain the same across
agents.

| Shared state                              | Why it is useful                                                    |
| ----------------------------------------- | ------------------------------------------------------------------- |
| Task constraints                          | Expose incompatible dates, versions, entities, and required outputs |
| Observations with provenance              | Reuse evidence without counting paraphrases as new observations     |
| Candidate status and unresolved relations | Separate plausible routes from accepted conclusions                 |
| Active reservations and dependencies      | Avoid conflicting work and budget for multi-step acquisition        |
| Accepted artifacts and their versions     | Integrate results against an explicit acceptance contract           |

An illustrative loop is short:

```text
read the current task and relevant shared state
propose an action or dependent action bundle
check that its reservation is still valid
execute within the remaining budget
attach observations and claim-level provenance
apply acceptance checks and update the shared state
```

This is a protocol sketch, not a claim that its semantic components have been
solved. “Propose a useful action” still requires judgment. “Apply acceptance
checks” still depends on the task's verifier. The point is to make those
interfaces observable and replaceable, rather than hide them inside a
personality description.

The communication structure should follow the work. Independent document
partitions can be explored separately before aggregation. Competing updates
to the same resource need coordinated commits. A multi-hop evidence chain
needs dependency-aware acquisition. A tightly coupled implementation task may
need a shared interface contract and frequent integration.

Permanent all-to-all broadcasting, fixed isolated teams, and a deep hierarchy
are all too coarse to answer these different needs with one choice. The
right unit of design is often an operation and its dependencies, rather than
the agent as a permanent organizational node.

Two baseline lessons keep this proposal honest.

First, static partitioning can already solve some coordination problems. If
one useful item is uniformly hidden among $N$ locations and the team can
inspect $B$ distinct locations without other clues, the best evidence-finding
probability is $B/N$. A preassigned disjoint partition can attain it. A dynamic
swarm cannot claim credit merely for avoiding duplicates that a static
assignment also avoids.

Second, literal duplicate queries may not be the actual waste. In 24 earlier
retrieval traces, 208 requests contained no exact duplicate requests, while
157 of 832 returned passage occurrences had already appeared within their
task. Ten queries returned only previously seen passages. That does not mean
18.87% of total tokens were avoidable: query outcomes were unknown in advance,
passages differed in length, and the rest of the prompt had its own cost.
The problem was overlapping information acquisition, which is harder to
predict than identical request strings.

### A stronger next experiment

I would choose a bounded task family with independently checkable outcomes,
freeze the utility assigned to errors and abstention, and compare the proposed
protocol against a strong sequential agent, static disjoint allocation, and
shared-index retrieval. The comparison should account for full inference
usage, tool costs, and end-to-end latency, including any controller or
workflow-search overhead.

The most informative task manipulation would vary dependency structure:
separable coverage, substitutable evidence, complementary evidence pairs,
and conflicting writes. It would test whether the proposed organization
helps in the regime its mechanism predicts. Policy selection would stay on
development data, followed by new held-out tasks and repeated runs.

That would turn “this organizational chart feels intelligent” into a claim
about an identifiable mechanism, a specified objective, and a measured
operating regime.

## 14. Evidence and reproducibility

The main numerical claims in this post come from the following distinct
stages. The exploratory stages should not be pooled into one apparent sample
of independent tasks.

| Stage                                    | Scope                                       | Main result or purpose                                                    |
| ---------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------- |
| Coverage and visibility mathematics      | Finite exact checks plus analytic arguments | Conditional alignment and equilibrium bounds; explicit cycles             |
| Provenance representation microbenchmark | 24 paired cases, 48 model calls             | Raw 24/24 versus board 23/24; lower board token usage                     |
| Real visibility intervention             | 8 fixed questions × 4 policies              | P/S 1/8, E/H 2/8; E used more tokens                                      |
| Fixed-input synthesis check              | One post hoc question, 12 calls             | Correctness varied even with unchanged supplied evidence                  |
| Relation controls                        | 10 post hoc cases × 2 calls                 | 20/20 support labels; two rejected-case explanations failed exact quoting |
| Budgeted final action                    | 8 reused questions, B/V twice and N once    | B 3/16 correct, V 4/16 correct with one wrong submission                  |
| Complementary-query counterexample       | 24 exact Bayesian instances                 | One-step decision-value greedy can have a ratio approaching zero          |

An earlier checkpoint pilot also failed to establish an advantage for hiding
the existing answer during recomputation: reuse and blind recomputation each
produced 15/42 correct outputs, while fresh exploration produced 16–17/42
depending on the judge. It motivated controls; it is not confirmatory support
for information isolation.

The visibility study and its subsequent synthesis, relation, and final-action
controls recorded **4,413,767 input-plus-output tokens**, including grading.
This includes cached input in the reported usage and excludes the older
pilot and provenance microbenchmark. It is not a dollar estimate. The final
action stage alone recorded 1,621,813 new collection-and-grading tokens.
Complete-policy totals separately charge each policy for reused prefixes;
those totals must not be mistaken for newly incurred collection usage.

The public companion files provide
[aggregate measurements and compact per-trajectory outcomes](/data/agent-swarm-organization/results.json),
a [runnable exact-theory check](/data/agent-swarm-organization/check_theory.py),
and [scope and reproduction notes](/data/agent-swarm-organization/README.md).
The companion script rechecks the stated small mathematical examples and
resource-pattern certificates using the Python standard library. It does
not rerun LLM inference or reproduce the entire original enumeration.

The public extract contains no source-corpus redistribution or raw model
sessions. Readers can recompute the published counts and small mathematical
examples; the extract alone cannot independently replay the complete LLM
collection. The source corpus and official retrieval implementation are
available from BrowseComp-Plus. Exact model-output replication is additionally
limited by the unpinned provider snapshot and unavailable sampling seed.

## The design question that remains

A productive agent swarm needs a causal connection between local action and
shared progress. In a coverage game, marginal contribution supplies that
connection exactly. Accurate resource counts make a small information
interface sufficient. Provenance union prevents a repeated observation from
turning into fictitious additional evidence.

Open-ended work stretches each assumption. The value of a query may be
unknown until execution. Two individually unhelpful observations may solve
the task together. A candidate can be correct without its supporting
relationship being present. A verifier can accept a persuasive statement
whose premises do not establish it.

The most useful organization is therefore one whose information flow,
dependencies, and acceptance conditions match the task well enough that
individual work can be measured and integrated. Establishing that match is
the research problem. Giving the agents more elaborate titles does not
establish it.
