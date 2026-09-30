# Agentic Deep Research

<aside>
💡

**Welcome!** Everything you need should be covered here. The team is a resource for you — feel free to pull us aside to chat through ideas, ask questions, or share updates.

</aside>

# Init

*Project: Agentic deep research over code*

Goal: RL-train a small LLM to be a maximally efficient code Q&A agent, and serve it in a full-stack product that gives deep-research-style answers to any question about any codebase.

At a high level, you’ll need to:

- Make the dataset
- Train the model
- Build the product

### Product

At a high level, the functionality we want to support is:

- User: selects a repo and types/dictates a question
- Agent: explores the codebase and writes a response

The UX and implementation are completely up to you. Be creative and have fun with it! What might we do differently compared to a general-audience chatbot?

During RL, you get what you grade for. Given that we’re going to train a model specifically for this use case, it’s useful to think through:

- How the agent will be implemented
    - How will the task be presented to the agent (system prompt, user prompt, etc.)?
    - What tool(s) will the agent need?
    - How will the tool(s) be implemented?
- What a “good” answer is in this context
    - Put differently: given a question and two candidate answers, what makes one better?
    - How can we grade for what we want during training?
    - How can we grade defensively, to ensure that our model can’t find ways to make the score go up without qualitatively improving at the task (i.e. reward hacking)?

### Dataset

An RL dataset is a collection of tasks. Concretely, for LLM RL this is often a JSONL file or some analogous thing. 

A task broadly defined is anything that we want our policy to do. More usefully, an RL task generally consists of:

- What we want the model to do
    - Often this is either a complete prompt, or whatever arguments are needed to parameterize a prompt template
- Whatever is required to set up the environment for this task
    - For a legal task, this might be a ref to a Docker image containing some Google Docs, PDFs, etc.
    - For a code-related task, this could be:
        - A path to a gzipped codebase in S3 (if all we need is the code)
        - A ref to a Docker image with a codebase and all the dependencies installed (if we want the model to be able to run code/tests)
        - An identifier for a commit that we want to check out
- Whatever is required to grade an attempt at this task
    - Examples:
        - A verifiable answer
            - e.g. `42` if the task is “What’s `21*2`?”
        - A rubric of task-specific criteria that will be used for LLM judging
            - Could be represented as a list of strings, or something more exotic if, for example, the criteria are weighted or gated
        - A “reference answer” for an LLM judge to compare the candidate answer against
    - There’s endless possibility here — much of the creativity in doing RL for a new task is figuring out (i) what “good” means (ii) how to make the dataset (iii) how to grade

**Building the Dataset**

Time to brainstorm:

- What does a single “task” instance look like in our context?
- How can we build a sufficiently large (e.g. 1,000-5,000-example) dataset of such tasks?

Feel free to chat through your ideas!

### Training

Given a dataset of tasks, LLM RL training consists of:

- Having our model attempt each task many times
- Positively reinforcing good behavior, and negatively reinforcing bad
- Aside: the meaning of “good” and “bad”
    
    The prevailing way to define “good” and “bad” in LLM RL are simply “better-than-average” and “worse-than-average” — given a group $G = \{o_1, \dots, o_n\}$ of $n$ outputs for the same task, we:
    
    - Grade each attempt independently to produce a **reward** $r_i$ (typically in $\{0,1\}$ or $[0,1]$)
    - Compute the mean $\mu = \frac{1}{n}\sum_i r_i$
    - Compute a per-attempt **advantage** $A_i = r_i - \mu$
        - Intuitively, this is a “goodness” score (which can be negative) and represents how much we want to reinforce the behavior that produced this output

To get started quickly, we suggest using the Tinker training API. Other options include open-source RL training frameworks like miles, veRL, and trl, as well as writing your own RL framework. Any of these could run on Modal or a GPU cluster from elsewhere. If you’re familiar with any particular option, feel free to use that, or use this opportunity to try something new!

We suggest using a very small model (e.g. Qwen3.5-9B or even Qwen3.5-4B). This is fun because:

- You can see model behavior change fairly quickly during training
- Inference will be really fast when you drop it into the product
- Tiny models can get pretty good when specialized effectively (good data and training recipe)

If you want really fast iteration cycles early on, you could run initial experiments on something even smaller.

Note that we want to push on the “efficiency” axis. How might we do that? What does it mean to do this task efficiently, and how can we create optimization pressure for that?

To support rigorous and principled iteration during training, you’ll want to set up at least the following:

- Logging of training metrics to something like WandB (or similar, if you have a preference)
- A way to see:
    - The outputs your model is generating (a.k.a “rollouts” or “trajectories”)
    - How your grader is scoring those attempts
    - How these things are changing over the course of training
        - For example, on your held-out eval set, which you could evaluate on every $n$ steps (e.g. $n = 10$)
    - Whatever else is useful for you to understand model behavior
        - Be creative! There’s a lot of possibility here

What we’re optimizing for is a reward curve that goes up over the course of training, and model outputs that pass the vibe check (up to you what this means in this context, and if there are additional metrics you can log to make this rigorous).

Iteration could consist of things like:

- Tweaking training hyperparameters to get fast, stable learning
- Experimenting with different approaches to grading (a.k.a “reward shaping”)
    - Even if your first attempt produces reasonable learning, you may want to try a couple of wildly different approaches just to see how you can shape model behavior
- Changing prompts or prompt templates (or other kinds of dataset pre-processing)
- Changing tool schemas, adding/removing tools, etc.

### Putting it All Together

You should be able to get the weights from Tinker or wherever you’re saving checkpoints, spin up an inference server somewhere (e.g. Modal), point your product at it, and let your tiny deep research agent rip.

### Recommended Resources

- Modal for dataset preparation and for hosting task environments during training (sandboxes)
- Tinker API for hosted RL training
- Token APIs (OpenAI, Anthropic, Fireworks, Baseten, etc.)
- Your chosen coding agent(s)
- Us!
    - Feel free to ask or chat about anything along the way
        - Project-specific ideas
        - RL in the wild (systems, infra, algorithms, data, etc.)
        - Other deep learning concepts/curiosities
        - Product and UX ideas
        - Anything else that comes up
- Ramp card
    
    ```jsx
    
    ```
    

### Deliverables

- Code
- Model checkpoint(s)
- A ~15-minute presentation covering:
    - What you did
    - How you chose to sequence work and manage your time
    - What you learned
    - What you’d do differently next time
    - How your thinking about the project evolved while working on it
    - What you’d do next

Good luck, and have fun! Feel free to try to get the model to do something interesting, like:

- Output citations with perfect format adherence
- Use an LSP effectively
- ???