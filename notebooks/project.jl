### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ f6d4a5e0-c004-11ef-0011-1a2b3c4d5e6f
begin
    using Plots

    # ── SIR forward-Euler integrator ────────────────────────────
    function sir_euler(β, γ, N, T; substeps=1)
        dt = 1.0 / substeps
        S = zeros(T); I = zeros(T); R = zeros(T)
        S[1] = N - 1.0;  I[1] = 1.0;  R[1] = 0.0
        for t in 1:T-1
            s, i, r = S[t], I[t], R[t]
            for _ in 1:substeps
                new_inf = β * i * s / N * dt
                new_rec = γ * i * dt
                s += -new_inf
                i +=  new_inf - new_rec
                r +=  new_rec
            end
            S[t+1] = s;  I[t+1] = i;  R[t+1] = r
        end
        S, I, R
    end

    # ── Final attack rate via Newton's method ───────────────────
    function attack_rate(R₀)
        R₀ ≤ 1 && return 0.0
        z = 1 - 1/R₀
        for _ in 1:200
            f  =  1 - z - exp(-R₀ * z)
            df = -1 + R₀ * exp(-R₀ * z)
            z -= f / df
        end
        clamp(z, 0.0, 1.0)
    end

    N = 1000;  γ = 0.1;  T = 200

    # ── Panel 1: I(t) for three regimes ─────────────────────────
    scenarios = [
        (0.5, "R₀ = 0.5  (sub-critical)",  :steelblue),
        (1.0, "R₀ = 1.0  (critical)",       :goldenrod),
        (2.5, "R₀ = 2.5  (super-critical)", :crimson),
    ]

    p1 = plot(
        title="Infectious individuals I(t) under three R₀ regimes",
        xlabel="Time (days)", ylabel="I(t)",
        legend=:topright, lw=2.5, background_color=:white,
        titlefontsize=11,
        left_margin=6Plots.mm, bottom_margin=5Plots.mm
    )
    for (R₀, label, col) in scenarios
        _, I, _ = sir_euler(R₀ * γ, γ, N, T)
        plot!(p1, 1:T, I; label, color=col)
    end

    p1
end

# ╔═╡ e5c3f4d0-bf03-11ef-0010-1a2b3c4d5e6f
begin
    using CSV, DataFrames

    raw = CSV.read(
        joinpath(@__DIR__, "data", "boarding_school_1978.csv"),
        DataFrame
    )

    # Convert dates to day indices (day 1 = Jan 22, 1978)
    data = DataFrame(
        day     = 1:nrow(raw),
        date    = raw.date,
        in_bed  = raw.in_bed,
    )
end

# ╔═╡ b8f6e3d0-e206-11ef-0015-1a2b3c4d5e6f
begin
    using Random

    # ── Gillespie direct method for SIR ─────────────────────────
    function gillespie_sir(β, γ, N, rng=MersenneTwister(); i₀=1)
        s, i = N - i₀, i₀
        t    = 0.0
        ts   = Float64[0.0]
        Is   = Int[i]
        while i > 0
            λ_inf = β * i * s / N
            λ_rec = γ * i
            λ_tot = λ_inf + λ_rec
            t    += randexp(rng) / λ_tot
            if rand(rng) < λ_inf / λ_tot
                s -= 1;  i += 1   # infection
            else
                i -= 1            # recovery
            end
            push!(ts, t);  push!(Is, i)
        end
        ts, Is, N - s   # (event times, I-path, final attack count)
    end

    # ── Parameters: 1978 boarding school influenza outbreak ──────
    N_g         = 763    # total boys
    R₀_g        = 2.5   # influenza estimate (to be fitted later)
    γ_g         = 0.5   # recovery rate (~2-day infectious period)
    β_g         = R₀_g * γ_g
    i₀_g        = 1     # single introduced case
    n_overlay   = 30
    n_histogram = 400
    fizzle_cut  = 0.05 * N_g

    # ── Panel 1: overlaid sample paths ──────────────────────────
    ov_paths = [gillespie_sir(β_g, γ_g, N_g, MersenneTwister(k); i₀=i₀_g) for k in 1:n_overlay]

    p_paths = plot(
        title="Gillespie sample paths  (R₀ = $(R₀_g),  N = $(N_g),  i₀ = $(i₀_g))",
        xlabel="Time (days)", ylabel="Infectious  i(t)",
        legend=:topright, background_color=:white, titlefontsize=11
    )

    _, I_det, _ = sir_euler(β_g, γ_g, N_g, 500)
    plot!(p_paths, 1:500, I_det;
        color=:black, lw=2.5, linestyle=:dash, label="Deterministic ODE")

    showed_fizzle   = Ref(false)
    showed_epidemic = Ref(false)
    for (ts_k, Is_k, final_sz) in ov_paths
        fizzled = final_sz < fizzle_cut
        col = fizzled ? :steelblue : :crimson
        α   = fizzled ? 0.50 : 0.55
        lbl = ""
        if fizzled && !showed_fizzle[]
            lbl = "Fizzled";  showed_fizzle[] = true
        elseif !fizzled && !showed_epidemic[]
            lbl = "Epidemic";  showed_epidemic[] = true
        end
        plot!(p_paths, ts_k, Is_k; color=col, alpha=α, lw=1.2, label=lbl)
    end

    # ── Panel 2: histogram of final attack rates ─────────────────
    his_paths         = [gillespie_sir(β_g, γ_g, N_g, MersenneTwister(k + 1000); i₀=i₀_g) for k in 1:n_histogram]
    final_pcts        = [sz / N_g * 100 for (_, _, sz) in his_paths]
    z_star_pct        = attack_rate(R₀_g) * 100
    fizzle_theory_n   = round(Int, n_histogram * (1/R₀_g)^i₀_g)

    p_hist = histogram(
        final_pcts;
        bins=range(0, 100, step=3),
        title="Final attack rate — $n_histogram runs  (R₀ = $(R₀_g),  N = $(N_g))",
        xlabel="Final attack rate  z (%)", ylabel="Count",
        color=:steelblue, alpha=0.7,
        legend=:topright, label="Simulations",
        background_color=:white, titlefontsize=11
    )
    vline!(p_hist, [z_star_pct];
        color=:black, lw=2.0, linestyle=:dash, label="z* (theory)")
    annotate!(p_hist, 1.5, fizzle_theory_n * 0.55,
        text("≈$(round(Int, (1/R₀_g)^i₀_g * 100))% fizzle\n(1/R₀)^{i₀}", :steelblue, :left, 8))

    plot(p_paths, p_hist; layout=(2, 1), size=(720, 660))
end

# ╔═╡ d1e2f3a4-0018-11ef-0018-1a2b3c4d5e6f
begin
    using Distributions

    # pgf: Neg-Binom offspring (mean R₀, dispersion k); Poisson limit when k = Inf
    bp_pgf(q, R₀, k) = isinf(k) ? exp(R₀*(q - 1)) : (1 + (R₀/k)*(1 - q))^(-k)

    # Fixed-point iteration q ← G(q) starting from q = 0
    function solve_q(R₀, k; tol=1e-12, maxiter=5000)
        q = 0.0
        for _ in 1:maxiter
            qn = bp_pgf(q, R₀, k)
            abs(qn - q) < tol && return qn
            q = qn
        end
        q
    end

    # Single branching-process trial; true = chain went extinct
    function bp_run(R₀, k; threshold=1000, rng=MersenneTwister())
        dist = isinf(k) ? Poisson(R₀) : NegativeBinomial(k, k / (k + R₀))
        gen = 1;  total = 1
        while gen > 0 && total < threshold
            gen > 10 * threshold && return false   # clearly escaped
            gen   = sum(rand(rng, dist) for _ in 1:gen)
            total += gen
        end
        total < threshold
    end

    # Estimate q from n_sim independent trials (fixed seed for reproducibility)
    function sim_q(R₀, k, n_sim=10_000; seed=42)
        rng0 = MersenneTwister(seed)
        count(bp_run(R₀, k; rng=MersenneTwister(rand(rng0, UInt32))) for _ in 1:n_sim) / n_sim
    end

    nothing
end

# ╔═╡ d1e2f3a4-0021-11ef-0021-1a2b3c4d5e6f
begin
    using PlutoUI
    md"**k** (dispersion)  |  **R₀**:"
end

# ╔═╡ c0d1e2f3-0300-11ef-0300-1a2b3c4d5e6f
using Graphs, LinearAlgebra

# ╔═╡ c3a1f2a0-9e01-11ef-0001-1a2b3c4d5e6f
html"""
<script>
(function () {
  const CSS = `
    h1 { color: #ffffff !important; font-family: Georgia, serif; font-size: 2.1rem;
         border-bottom: 3px solid #ffffff; padding-bottom: 0.3em; }
    h2 { color: #ffffff !important; font-family: Georgia, serif; font-size: 1.55rem;
         border-left: 5px solid #aaaaaa; padding-left: 0.6em; margin-top: 1.4em; }
    h3 { color: #ffffff !important; font-size: 1.2rem; }
    p, li { color: #ffffff !important; font-size: 1.05rem; line-height: 1.65; }
    table { border-collapse: collapse; width: 100%; margin-top: 0.8em; }
    th { background-color: #333333 !important; color: #ffffff !important; padding: 9px 15px; text-align: left; }
    td { padding: 8px 15px; border-bottom: 1px solid #555555; color: #ffffff !important; }
    tr:nth-child(even) td { background-color: #2a2a2a !important; }
  `;

  // Inject into document head
  let g = document.getElementById("proj-global");
  if (g) g.remove();
  g = document.createElement("style");
  g.id = "proj-global";
  g.textContent = CSS;
  document.head.appendChild(g);

  // Inject into a shadow root (creates/updates a <style> tag inside it)
  function injectShadow(root) {
    let s = root.getElementById ? root.getElementById("proj-shadow")
                                : root.querySelector("#proj-shadow");
    if (!s) { s = document.createElement("style"); s.id = "proj-shadow"; root.appendChild(s); }
    s.textContent = CSS;
  }

  // Process every pluto-output and pluto-cell that has a shadow root
  function processAll() {
    ["pluto-output", "pluto-cell"].forEach(tag => {
      document.querySelectorAll(tag).forEach(el => {
        if (el.shadowRoot) injectShadow(el.shadowRoot);
      });
    });
  }

  processAll();

  // Re-run whenever new cells are added (reactive notebook)
  if (!window._projObserver) {
    window._projObserver = new MutationObserver(processAll);
    window._projObserver.observe(document.body, { childList: true, subtree: true });
  }
})();
</script>
"""

# ╔═╡ c3a1f2a0-9e01-11ef-0002-1a2b3c4d5e6f
md"""
# Infectious Models of Disease

**18.C25 — Class Presentation** · *Fall 2026*
"""

# ╔═╡ c3a1f2a0-9e01-11ef-0003-1a2b3c4d5e6f
md"""
## Section 1: The SIR Model

The SIR model is the canonical compartmental model in mathematical epidemiology.
It partitions a **closed, fixed** population of size $N$ into three mutually exclusive groups:

| Compartment | Symbol | Meaning |
|:---|:---:|:---|
| **Susceptible** | $S_t$ | Individuals who have not yet been infected and can contract the disease |
| **Infectious** | $I_t$ | Individuals who are currently infected and can transmit the disease |
| **Recovered** | $R_t$ | Individuals who have recovered and are permanently immune |

**Conservation law:** $\quad S_t + I_t + R_t = N \quad$ for all $t \geq 0$.
"""

# ╔═╡ c3a1f2a0-9e01-11ef-0005-1a2b3c4d5e6f
md"""
### Parameters

| Symbol | Name | Description |
|:---:|:---|:---|
| $\beta > 0$ | Transmission rate | Rate of effective contact between susceptible and infectious individuals |
| $\gamma > 0$ | Recovery rate | Rate at which infectious individuals recover and become immune |
| $N$ | Population size | Total (constant) population size: $N = S_t + I_t + R_t$ |
| $R_0 = \dfrac{\beta}{\gamma}$ | Basic reproduction number | Expected number of secondary infections caused by one infectious individual in a fully susceptible population |
"""

# ╔═╡ c3a1f2a0-9e01-11ef-0004-1a2b3c4d5e6f
md"""
### System of Differential Equations

$$\begin{aligned}
\dfrac{dS_t}{dt} &= -\dfrac{\beta \, I_t \, S_t}{N} \\[12pt]
\dfrac{dI_t}{dt} &= \phantom{-}\dfrac{\beta \, I_t \, S_t}{N} - \gamma \, I_t \\[12pt]
\dfrac{dR_t}{dt} &= \phantom{-}\gamma \, I_t
\end{aligned}$$
"""

# ╔═╡ d4b2e3c0-ae02-11ef-0006-1a2b3c4d5e6f
md"""
### Early Epidemic Approximation

At the very start of an epidemic, nearly everyone is susceptible: $S_0 \approx N$, so $S_0/N \approx 1$. The equation for $I_t$ simplifies to:

$$\frac{dI_t}{dt} \approx (\beta - \gamma)\, I_0$$

This is a linear ODE with closed-form solution:

$$\boxed{I(t) \;\approx\; I_0\, e^{(\beta - \gamma)\,t} \;=\; I_0\, e^{\,\gamma(R_0 - 1)\,t}}$$

The sign of $\beta - \gamma$ — equivalently, whether $R_0 \gtrless 1$ — determines the fate of the epidemic.
"""

# ╔═╡ d4b2e3c0-ae02-11ef-0008-1a2b3c4d5e6f
md"""
### Final Attack Rate

Even when $R_0 > 1$ and $I(t)$ grows exponentially at first, the epidemic cannot continue indefinitely. As infections accumulate, the susceptible pool $S_t$ depletes, weakening the force of infection — eventually recovery outpaces transmission and $I(t) \to 0$.

The fraction of the population **ever infected**, $z = 1 - S_\infty / N$, satisfies the implicit equation:

$$\boxed{1 - z = e^{-R_0\, z}}$$

This has a unique positive solution only when $R_0 > 1$, and $z = 0$ when $R_0 \leq 1$ — a **phase transition** at $R_0 = 1$.
"""

# ╔═╡ d4b2e3c0-ae02-11ef-0009-1a2b3c4d5e6f
let
    R₀_vals = 0.0:0.005:4.0
    z_vals   = attack_rate.(R₀_vals)

    fig = plot(R₀_vals, z_vals;
        xlabel="R₀  (basic reproduction number)",
        ylabel="z  (final attack rate)",
        title="Phase transition: z is nonzero if and only if R₀ > 1",
        lw=2.5, color=:crimson, legend=false,
        titlefontsize=11,
        ylims=(-0.03, 1.05), xlims=(0, 4),
        background_color=:transparent, foreground_color=:white,
        left_margin=6Plots.mm, bottom_margin=5Plots.mm,
        size=(640, 340)
    )

    vspan!(fig, [0.0, 1.0]; color=:steelblue, alpha=0.12, label="")
    vline!(fig, [1.0]; color=:white, lw=1.5, linestyle=:dash, label="")
    annotate!(fig, 1.07, 0.97, text("R₀ = 1", :white, :left, 9))
    annotate!(fig, 0.5,  0.55, text("z = 0\n(no epidemic)", :white, :center, 10))
    annotate!(fig, 2.7,  0.30, text("z > 0\n(epidemic spreads)", :crimson, :center, 10))
    scatter!(fig, [1.0], [0.0]; color=:white, markersize=6, markerstrokecolor=:white, label="")
    fig
end

# ╔═╡ d4b2e3c0-ae02-11ef-0007-1a2b3c4d5e6f
md"""
### $R_0$ as a Threshold

| $R_0$ | Regime | Early behavior of $I(t)$ | Final attack rate $z$ |
|:---:|:---:|:---|:---:|
| $R_0 < 1$ | Sub-critical | Exponential **decay** — epidemic fizzles out | $z \approx 0$ |
| $R_0 = 1$ | Critical | Neither growth nor decay | $z \to 0^+$ |
| $R_0 > 1$ | Super-critical | Exponential **growth** — epidemic takes off | $z > 0$ |
"""

# ╔═╡ e1f2a3b4-0028-11ef-0028-1a2b3c4d5e6f
md"**Explore the SIR model** — move the sliders to see how R₀ and the infectious period shape the epidemic:"

# ╔═╡ e1f2a3b4-0029-11ef-0029-1a2b3c4d5e6f
@bind R₀_live Slider(0.5:0.1:4.0; default=2.5, show_value=true)

# ╔═╡ e1f2a3b4-0030-11ef-0030-1a2b3c4d5e6f
@bind γ_live Slider(0.05:0.05:1.0; default=0.1, show_value=true)

# ╔═╡ e1f2a3b4-0031-11ef-0031-1a2b3c4d5e6f
let
    β_live = R₀_live * γ_live
    N_live = 1000
    T_live = min(round(Int, 8 / γ_live + 50), 2000)

    S, I_c, R_c = sir_euler(β_live, γ_live, N_live, T_live; substeps=10)
    peak_I = maximum(I_c)
    peak_t = argmax(I_c)
    z_live = attack_rate(R₀_live)

    fig = plot(1:T_live, S;
        color=:steelblue, lw=2, label="S(t)  susceptible",
        xlabel="Time (days)", ylabel="Individuals",
        title="R₀ = $R₀_live   |   γ = $γ_live   (infectious period ≈ $(round(1/γ_live, digits=1)) days)",
        titlefontsize=11, legend=:right, size=(780, 380),
        background_color=:transparent, foreground_color=:white,
        left_margin=6Plots.mm, bottom_margin=5Plots.mm)
    plot!(fig, 1:T_live, I_c; color=:crimson,   lw=2.5, label="I(t)  infectious")
    plot!(fig, 1:T_live, R_c; color=:goldenrod, lw=2,   label="R(t)  recovered")

    if peak_I > 5
        vline!(fig, [peak_t]; color=:white, lw=1, linestyle=:dash, label="")
        annotate!(fig, peak_t + max(T_live*0.02, 2), peak_I * 0.93,
            text("peak I = $(round(Int, peak_I))  (day $peak_t)", :white, :left, 9))
    end
    annotate!(fig, T_live * 0.62, N_live * 0.18,
        text("final attack rate z ≈ $(round(z_live*100, digits=1))%", :white, :left, 10))
    fig
end

# ╔═╡ a0b1c2d3-0410-11ef-0410-1a2b3c4d5e6f
md"""
### Fitting to Data: 1978 Boarding School Influenza

763 boys in a closed boarding school; daily *in-bed* counts over 14 days. Move the sliders to fit the deterministic SIR curve by eye — the SSE updates live. The ODE starts at $I=1$; for each $(\gamma, R_0)$ the best time alignment $t_0$ is found automatically.
"""

# ╔═╡ a0b1c2d3-0412-11ef-0412-1a2b3c4d5e6f
@bind R₀_fit Slider(1.0:0.1:5.0; default=2.5, show_value=true)

# ╔═╡ a0b1c2d3-0413-11ef-0413-1a2b3c4d5e6f
@bind γ_fit Slider(0.05:0.05:1.5; default=0.5, show_value=true)

# ╔═╡ a0b1c2d3-0414-11ef-0414-1a2b3c4d5e6f
let
    N   = 763
    obs = Float64.(data.in_bed)
    n   = length(obs)

    β_fit = R₀_fit * γ_fit
    _, I_sim, _ = sir_euler(β_fit, γ_fit, N, 80; substeps=20)

    # Best time alignment for current (R₀, γ)
    best_t₀  = 1
    best_sse = Inf
    for t₀ in 1:80-n
        sse = sum((I_sim[t₀ + k - 1] - obs[k])^2 for k in 1:n)
        if sse < best_sse
            best_sse = sse
            best_t₀  = t₀
        end
    end

    t_end  = min(80, best_t₀ + n + 3)
    I_show = I_sim[best_t₀ : t_end]

    fig = scatter(1:n, obs;
        color=:white, ms=5, markerstrokecolor=:white,
        label="Observed (in bed)",
        xlabel="Day", ylabel="Students in bed  I(t)",
        title="R₀ = $R₀_fit   γ = $γ_fit   (inf. period ≈ $(round(1/γ_fit, digits=1)) days)",
        titlefontsize=11, legend=:topright,
        ylims=(0, 450),
        background_color=:transparent, foreground_color=:white,
        size=(760, 380), bottom_margin=8Plots.mm, left_margin=6Plots.mm)
    plot!(fig, 1:length(I_show), I_show;
        color=:crimson, lw=2.5, label="Deterministic SIR")
    annotate!(fig, 10.5, 415, text("SSE = $(round(Int, best_sse))", :white, :left, 13))
    fig
end

# ╔═╡ a7e5d2c0-d105-11ef-0012-1a2b3c4d5e6f
md"""
## Stochasticity and Early Extinction

### 1. Markov Chain Formulation

Instead of smooth flows, infection and recovery are **random events**, one person at a time. In a short window $\Delta t$:

| Event | State transition | Probability |
|:---|:---:|:---|
| Infection | $(s,\, i) \;\to\; (s-1,\; i+1)$ | $\dfrac{\beta\, i\, s}{N}\,\Delta t$ |
| Recovery  | $(s,\, i) \;\to\; (s,\;\; i-1)$   | $\gamma\, i\;\Delta t$ |

The next step depends only on the **current state** $(s, i)$ — not on any prior history — which is exactly the **Markov property**. The epidemic ends as soon as $i = 0$.
"""

# ╔═╡ a7e5d2c0-d105-11ef-0013-1a2b3c4d5e6f
md"""
### Deterministic vs. Stochastic Formulation

The two formulations share the same parameters $\beta$ and $\gamma$, but differ in what the variables represent and how transitions occur.

|  | **Deterministic ODE** | **Stochastic Markov chain** |
|:---|:---|:---|
| **Variables** | $S, I, R \in \mathbb{R}$ — smooth, continuous quantities | $s, i \in \mathbb{Z}_{\geq 0}$ — integer counts; realizations of random variables $S_t, I_t$ |
| **Transitions** | Continuous flows at fixed rates | Discrete jumps, one individual at a time |
| **Randomness** | None | Inherent — same $(s,i)$ can lead to different outcomes |

**Deterministic** — population-level, uppercase:

$$\begin{aligned}
\dfrac{dS}{dt} &= -\beta I \dfrac{S}{N} \\[8pt]
\dfrac{dI}{dt} &= \beta I \dfrac{S}{N} - \gamma I \\[8pt]
\dfrac{dR}{dt} &= \gamma I
\end{aligned}$$

**Stochastic** — individual-level, lowercase:

$$p_{(s,\,i),\,(s+k,\,i+j)}(\Delta t) \;=\; \begin{cases} \beta i \,\dfrac{s}{N}\,\Delta t + o(\Delta t), & (k,j) = (-1,\,+1) \\[8pt] \gamma i\,\Delta t + o(\Delta t), & (k,j) = (0,\,-1) \\[8pt] 1 - \!\left(\beta i \dfrac{s}{N} + \gamma i\right)\Delta t + o(\Delta t), & (k,j) = (0,\,0) \end{cases}$$
"""

# ╔═╡ b8f6e3d0-e206-11ef-0014-1a2b3c4d5e6f
md"""
### 2. Gillespie Algorithm and Sample Paths

To simulate the stochastic SIR process exactly:

1. **Waiting time** — draw $\tau \sim \operatorname{Exp}(\lambda_\text{inf} + \lambda_\text{rec})$, where $\lambda_\text{inf} = \beta i s / N$ and $\lambda_\text{rec} = \gamma i$
2. **Event selection** — infection with probability $\lambda_\text{inf}\,/\,(\lambda_\text{inf} + \lambda_\text{rec})$, recovery otherwise; update $(s, i)$ and advance $t \leftarrow t + \tau$

Running the same $R_0$ many times reveals that individual outbreaks are fundamentally unpredictable: some fizzle immediately (the first case recovers before transmitting), while others ignite a full epidemic. The deterministic ODE traces the *average* over epidemic trajectories — it cannot capture extinction.
"""

# ╔═╡ e1f2a3b4-0032-11ef-0032-1a2b3c4d5e6f
md"**Live Gillespie run** — slide R₀ to shift the odds; slide run # to see different random outcomes:"

# ╔═╡ e1f2a3b4-0033-11ef-0033-1a2b3c4d5e6f
@bind R₀_gill Slider(0.5:0.1:4.0; default=2.5, show_value=true)

# ╔═╡ e1f2a3b4-0034-11ef-0034-1a2b3c4d5e6f
@bind seed_gill Slider(1:200; default=1, show_value=true)

# ╔═╡ e1f2a3b4-0035-11ef-0035-1a2b3c4d5e6f
let
    γ_gill = 0.5
    β_gill = R₀_gill * γ_gill

    ts, Is, final_sz = gillespie_sir(β_gill, γ_gill, N_g, MersenneTwister(seed_gill); i₀=1)

    fizzled = final_sz < round(Int, 0.05 * N_g)
    col     = fizzled ? :steelblue : :crimson
    outcome = fizzled ? "FIZZLED — chain died out" :
                        "EPIDEMIC — $(round(Int, final_sz/N_g*100))% infected"

    T_det = max(ceil(Int, maximum(ts)) + 20, 120)
    _, I_det, _ = sir_euler(β_gill, γ_gill, N_g, T_det)
    y_max = max(maximum(I_det), maximum(Is)) * 1.35

    fig = plot(ts, Is;
        color=col, lw=2.5, label="run #$seed_gill",
        xlabel="Time (days)", ylabel="Infectious  i(t)",
        title="Gillespie  |  R₀ = $R₀_gill  |  N = $N_g  |  i₀ = 1",
        titlefontsize=12, legend=:topright, size=(780, 400),
        xlims=(0, T_det), ylims=(0, y_max),
        background_color=:transparent, foreground_color=:white,
        left_margin=6Plots.mm, bottom_margin=5Plots.mm)
    plot!(fig, 1:T_det, I_det;
        color=:white, lw=1.5, linestyle=:dash, label="Deterministic ODE")
    annotate!(fig, T_det * 0.05, y_max * 0.88,
        text(outcome, col, :left, 14))
    fig
end

# ╔═╡ c9e7b4f0-f307-11ef-0016-1a2b3c4d5e6f
md"""
### Fizzle Probability: The $(1/R_0)^{i_0}$ Rule

Near the start of an outbreak ($i \approx 0$, $s \approx N$), each infectious individual independently infects others at rate $\beta$ and recovers at rate $\gamma$ — so the infectious chain behaves like a **branching process** with offspring mean $R_0$.

For a supercritical branching process ($R_0 > 1$) started from $i_0$ independent cases, the probability that **every chain goes extinct** before taking off is:

$$\boxed{P(\text{fizzle}) = \left(\frac{1}{R_0}\right)^{i_0}}$$

Applied to the **1978 boarding school outbreak** with $R_0 \approx 2.5$ and a single introduced case ($i_0 = 1$):

$$P(\text{fizzle}) = \frac{1}{2.5} = 0.40 \qquad (40\%)$$

The histogram above shows this bimodality directly — roughly 40% of runs pile up near $z = 0$ (fizzled), while the rest cluster near the theoretical epidemic size $z^* \approx 89\%$. The outbreak we observe in the data is not inevitable: about 4 in 10 single-case introductions would have quietly died out.
"""

# ╔═╡ d1e2f3a4-0016-11ef-0016-1a2b3c4d5e6f
HTML("""
<div style="font-family: sans-serif; max-width: 660px; margin: 1.5em auto; color: #e0e0e0;">

  <!-- Shared assumptions -->
  <div style="background:#0d2340; border:2px solid #5baee0; border-radius:10px; padding:14px 20px; text-align:center;">
    <div style="font-weight:bold; font-size:1.05em; color:#7ec8f0; margin-bottom:6px;">Shared SIR assumptions</div>
    <div style="font-size:0.9em; line-height:1.7;">
      <b>β</b> identical for all &nbsp;·&nbsp; <b>γ</b> identical for all<br>
      infectious period ~ Exp(γ) &nbsp;→&nbsp; same rate, random duration per person<br>
      transmission rate β constant while infectious<br>
      offspring count ~ <b>Geometric</b> &nbsp;≡&nbsp; NegBinom(k = 1)
    </div>
  </div>

  <!-- Fork arrows -->
  <div style="display:flex; justify-content:space-around; padding:0 18%;">
    <div style="display:flex; flex-direction:column; align-items:center;">
      <div style="width:2px; height:16px; background:#666;"></div>
      <div style="color:#666; font-size:1.4em; line-height:1;">↓</div>
    </div>
    <div style="display:flex; flex-direction:column; align-items:center;">
      <div style="width:2px; height:16px; background:#666;"></div>
      <div style="color:#666; font-size:1.4em; line-height:1;">↓</div>
    </div>
  </div>

  <!-- ODE + Stochastic side by side -->
  <div style="display:flex; gap:14px;">
    <div style="flex:1; background:#0d2d1a; border:2px solid #4caf70; border-radius:10px; padding:12px 14px; text-align:center;">
      <div style="font-weight:bold; color:#7dd4a0; margin-bottom:4px;">Deterministic ODE</div>
      <div style="font-size:0.85em; line-height:1.6;">S, I, R ∈ ℝ<br>population-level flows</div>
    </div>
    <div style="flex:1; background:#2d0d1a; border:2px solid #e06070; border-radius:10px; padding:12px 14px; text-align:center;">
      <div style="font-weight:bold; color:#e88090; margin-bottom:4px;">Stochastic CTMC (Gillespie)</div>
      <div style="font-size:0.85em; line-height:1.6;">s, i, r ∈ ℤ<sub>≥0</sub><br>individual integer jumps</div>
    </div>
  </div>

  <!-- Relax arrow -->
  <div style="display:flex; flex-direction:column; align-items:center; margin:4px 0;">
    <div style="width:2px; height:14px; background:#666;"></div>
    <div style="color:#aaa; font-size:0.82em; font-style:italic; background:#1e1e2e; padding:3px 12px; border-radius:4px; border:1px solid #444;">
      relax: allow β, γ to vary by individual
    </div>
    <div style="color:#666; font-size:1.4em; line-height:1.3;">↓</div>
  </div>

  <!-- Branching process -->
  <div style="background:#2a1a0d; border:2px solid #e0a040; border-radius:10px; padding:14px 20px; text-align:center;">
    <div style="font-weight:bold; color:#e8b860; margin-bottom:6px;">Branching process — NegBinom(R₀, k)</div>
    <div style="font-size:0.9em; line-height:1.7;">
      k = 1 &nbsp;→&nbsp; geometric, recovers standard SIR &nbsp;(q* = 1/R₀)<br>
      k &lt; 1 &nbsp;→&nbsp; overdispersed: most infect few, rare superspreaders<br>
      k → ∞ &nbsp;→&nbsp; Poisson, maximally homogeneous
    </div>
  </div>

</div>
""")

# ╔═╡ d1e2f3a4-0017-11ef-0017-1a2b3c4d5e6f
md"""
### 3. Branching Processes and Superspreading

Each case independently draws its **offspring count** from a distribution with mean $R_0$ and dispersion $k$. The **extinction probability** $q^*$ is the smallest fixed point of

$$q = G(q), \qquad G(q) = \sum_{j=0}^{\infty} P(j)\, q^j$$
"""

# ╔═╡ d1e2f3a4-0017-11ef-00fa-1a2b3c4d5e6f
let
    col_x = [0.0, 1.4, 4.1, 6.9, 10.3]
    row_y = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    W, H  = col_x[end], row_y[1]

    fig = plot(xlims=(0,W), ylims=(0,H),
               framestyle=:none, legend=false,
               size=(1050, 460),
               background_color=:transparent,
               foreground_color=:white,
               xticks=nothing, yticks=nothing)

    # Colored header backgrounds
    plot!(fig, Shape([col_x[2],col_x[3],col_x[3],col_x[2]],
                     [row_y[2],row_y[2],row_y[1],row_y[1]]);
          fillcolor=:midnightblue, fillalpha=0.9, linecolor=:steelblue, linewidth=2, label="")
    plot!(fig, Shape([col_x[3],col_x[4],col_x[4],col_x[3]],
                     [row_y[2],row_y[2],row_y[1],row_y[1]]);
          fillcolor=:saddlebrown, fillalpha=0.6, linecolor=:goldenrod, linewidth=2, label="")
    plot!(fig, Shape([col_x[4],col_x[5],col_x[5],col_x[4]],
                     [row_y[2],row_y[2],row_y[1],row_y[1]]);
          fillcolor=:darkred, fillalpha=0.9, linecolor=:crimson, linewidth=2, label="")

    # Alternating row tints
    for ri in [2, 4, 6]
        y0, y1 = row_y[ri+1], row_y[ri]
        plot!(fig, Shape([col_x[1],col_x[end],col_x[end],col_x[1]],
                         [y0,y0,y1,y1]);
              fillcolor=:white, fillalpha=0.04, linewidth=0, label="")
    end

    # Grid lines
    for x in col_x
        plot!(fig, [x,x],[0.0,H]; color=:gray, lw=0.7, label="")
    end
    for y in row_y
        plot!(fig, [0.0,W],[y,y]; color=:gray, lw=0.7, label="")
    end

    cx = [(col_x[i]+col_x[i+1])/2 for i in 1:4]
    ry = [(row_y[i]+row_y[i+1])/2 for i in 1:6]

    # Header text
    annotate!(fig, cx[2], ry[1], text("Poisson  (k → ∞)",    :skyblue,   :center, 11))
    annotate!(fig, cx[3], ry[1], text("Geometric  (k = 1)",   :gold,      :center, 11))
    annotate!(fig, cx[4], ry[1], text("Neg. Binom.  (k < 1)", :lightpink, :center, 11))

    # Row labels
    for (i, lbl) in enumerate(["Story","Params","P(j)","G(q)","q*"])
        annotate!(fig, cx[1], ry[i+1], text(lbl, :gray, :center, 10))
    end

    # Story row
    annotate!(fig, cx[2], ry[2]+0.14, text("Fixed period;",          :white, :center, 9))
    annotate!(fig, cx[2], ry[2]-0.14, text("all equally infectious",  :white, :center, 9))
    annotate!(fig, cx[3], ry[2]+0.14, text("Exp. period, const. β;", :white, :center, 9))
    annotate!(fig, cx[3], ry[2]-0.14, text("standard SIR",            :white, :center, 9))
    annotate!(fig, cx[4], ry[2]+0.24, text("Heterogeneous rates;",   :white, :center, 9))
    annotate!(fig, cx[4], ry[2],      text("most mild,",             :white, :center, 9))
    annotate!(fig, cx[4], ry[2]-0.24, text("few superspreaders",     :white, :center, 9))

    # Params row
    annotate!(fig, cx[2], ry[3],      text("λ = β/γ = R₀",          :white, :center, 10))
    annotate!(fig, cx[3], ry[3]+0.14, text("p = γ/(β+γ)",           :white, :center, 10))
    annotate!(fig, cx[3], ry[3]-0.14, text("= 1/(1+R₀)",            :white, :center, 10))
    annotate!(fig, cx[4], ry[3]+0.14, text("mean = R₀,  shape = k", :white, :center, 9))
    annotate!(fig, cx[4], ry[3]-0.14, text("Var = R₀(1 + R₀/k)",   :white, :center, 9))

    # P(j) row
    annotate!(fig, cx[2], ry[4],      text("e^(-R₀) · R₀^j / j!",  :white, :center, 10))
    annotate!(fig, cx[3], ry[4],      text("R₀^j / (1+R₀)^(j+1)",  :white, :center, 10))
    annotate!(fig, cx[4], ry[4]+0.24, text("C(j+k-1, j)",           :white, :center, 9))
    annotate!(fig, cx[4], ry[4],      text("·(k/(k+R₀))^k",         :white, :center, 9))
    annotate!(fig, cx[4], ry[4]-0.24, text("·(R₀/(k+R₀))^j",       :white, :center, 9))

    # G(q) row
    annotate!(fig, cx[2], ry[5], text("exp(R₀(q-1))",               :white, :center, 10))
    annotate!(fig, cx[3], ry[5], text("1 / (1+R₀(1-q))",           :white, :center, 10))
    annotate!(fig, cx[4], ry[5], text("(1 + R₀(1-q)/k)^(-k)",      :white, :center, 10))

    # q* row
    annotate!(fig, cx[2], ry[6], text("numerical", :gray,  :center, 10))
    annotate!(fig, cx[3], ry[6], text("1/R₀",      :white, :center, 11))
    annotate!(fig, cx[4], ry[6], text("numerical", :gray,  :center, 10))

    fig
end

# ╔═╡ d1e2f3a4-0017-11ef-00fb-1a2b3c4d5e6f
md"""
**Boarding school** ($R_0 = 2.5$, geometric / standard SIR): substituting $G(q) = \frac{1}{1+R_0(1-q)}$ and rearranging gives

$$2.5\,q^2 - 3.5\,q + 1 = 0 \quad\Longrightarrow\quad q = 1 \;\text{ or }\; q^* = \tfrac{1}{2.5} = 0.40$$

The smallest root $q^* = 1/R_0 = 0.40$ is the **fizzle probability** — 40% of single-introduction outbreaks die out before takeoff. Smaller $k$ raises $q^*$ but makes surviving chains more explosive.
"""


# ╔═╡ d1e2f3a4-0019-11ef-0019-1a2b3c4d5e6f
let
    mkpath(joinpath(@__DIR__, "figures"))
    R₀_A = 2.0;  js = 0:15

    pmf_poi = pdf.(Poisson(R₀_A), js)
    pmf_geo = pdf.(NegativeBinomial(1,   1   / (1   + R₀_A)), js)
    pmf_nb  = pdf.(NegativeBinomial(0.1, 0.1 / (0.1 + R₀_A)), js)

    kw = (bar_width=0.65, lw=0, legend=false,
          xlabel="offspring  j", ylabel="P(j)",
          guidefontsize=14, tickfontsize=12, titlefontsize=15,
          background_color=:transparent, foreground_color=:white,
          bottom_margin=8Plots.mm, left_margin=6Plots.mm)

    ann(p0) = text("P(0) = $(round(p0, digits=2))", :white, :left, 12)

    fA1 = bar(js, pmf_poi; title="Poisson  (k → ∞)", color=:steelblue, kw...)
    annotate!(fA1, 0.5, maximum(pmf_poi)*0.88, ann(pmf_poi[1]))

    fA2 = bar(js, pmf_geo; title="Geometric  (k = 1)", color=:goldenrod, kw...)
    annotate!(fA2, 0.5, maximum(pmf_geo)*0.88, ann(pmf_geo[1]))

    fA3 = bar(js, pmf_nb;  title="Neg. Binom.  (k = 0.1)", color=:crimson, kw...)
    annotate!(fA3, 0.5, maximum(pmf_nb)*0.85,  ann(pmf_nb[1]))

    figA = plot(fA1, fA2, fA3; layout=(1,3), size=(1100, 440),
        plot_title="Offspring distributions  (R₀ = $R₀_A)",
        plot_titlefontsize=14)
    savefig(figA, joinpath(@__DIR__, "figures", "figA_offspring.png"))
    figA
end

# ╔═╡ d1e2f3a4-0020-11ef-0020-1a2b3c4d5e6f
md"""
#### Extinction Probability vs. Dispersion $k$

At the **same** $R_0$, smaller $k$ (more overdispersion) raises $q^*$: individual chains die out more easily because most people infect nobody. But when a chain *does* escape, a single superspreader can ignite a large cluster.

**SARS-CoV-1:** estimated $k \approx 0.16$ — 80% of transmission came from 20% of cases.

Use the slider below to explore $k$:
"""

# ╔═╡ d1e2f3a4-0022-11ef-0022-1a2b3c4d5e6f
@bind k_B Slider([0.05, 0.1, 0.16, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]; default=1.0, show_value=true)

# ╔═╡ d1e2f3a4-0024-11ef-0024-1a2b3c4d5e6f
let
    k_range = exp10.(range(log10(0.05), log10(10.0), length=200))

    figB = plot(;
        xlabel="Dispersion  k  (log scale)", ylabel="Extinction probability  q*",
        xscale=:log10, legend=:topright, lw=2.5, size=(820, 480),
        title="q* vs. dispersion k   (selected k = $k_B)",
        titlefontsize=13, guidefontsize=13, tickfontsize=11,
        background_color=:transparent, foreground_color=:white,
        bottom_margin=8Plots.mm, left_margin=6Plots.mm)

    for (R₀_c, col, lbl) in [(2.0, :steelblue, "R₀ = 2.0"),
                               (R₀_g, :crimson, "R₀ = $R₀_g  (school)")]
        plot!(figB, k_range, solve_q.(R₀_c, k_range); color=col, label=lbl)
    end

    for k_s in [0.1, 0.5, 1.0, 2.0, 5.0]
        qs = sim_q(2.0, k_s, 2000)
        se = sqrt(qs*(1-qs)/2000)
        scatter!(figB, [k_s], [qs]; yerr=1.96*se, color=:steelblue, ms=5, label="")
    end

    vline!(figB, [1.0];  color=:white,  lw=1.2, linestyle=:dash, label="k = 1  (SIR)")
    vline!(figB, [0.16]; color=:orange, lw=1.2, linestyle=:dash, label="k ≈ 0.16  (SARS)")
    scatter!(figB, [k_B], [solve_q(2.0, k_B)];
        color=:yellow, ms=9, markershape=:star5, label="selected  (R₀ = 2.0)")

    savefig(figB, joinpath(@__DIR__, "figures", "figB_extinction.png"))
    figB
end

# ╔═╡ d1e2f3a4-0025-11ef-0025-1a2b3c4d5e6f
md"""
#### Final Attack Rate  z  given Major Outbreak

Given that the chain *does* take off, what fraction of the population ultimately gets infected? The generalized final-size equation (Gamma-distributed individual $R$, mean $R_0$, shape $k$) is

$$z = 1 - \left(1 + \frac{R_0\,z}{k}\right)^{-k}, \qquad k \to \infty:\; z = 1 - e^{-R_0 z}$$

Smaller $k$ means superspreaders exhaust the susceptible pool faster, raising $z$ at the same $R_0$.
"""

# ╔═╡ d1e2f3a4-0026-11ef-0026-1a2b3c4d5e6f
let
    function final_size(R₀, k)
        R₀ ≤ 1 && return 0.0
        z = 1 - 1/R₀
        for _ in 1:400
            if isinf(k)
                f  = z - 1 + exp(-R₀*z)
                df = 1 - R₀*exp(-R₀*z)
            else
                u  = 1 + R₀*z/k
                f  = z - 1 + u^(-k)
                df = 1 - R₀*u^(-k-1)
            end
            z = clamp(z - f/df, 0.0, 1.0)
        end
        clamp(z, 0.0, 1.0)
    end

    R₀_range = 1.0:0.01:4.0

    fig = plot(;
        xlabel="R₀  (basic reproduction number)",
        ylabel="Final attack rate  z",
        legend=:topleft, lw=2.5, size=(760, 460),
        title="Final attack rate given major outbreak, by dispersion k",
        titlefontsize=13, guidefontsize=13, tickfontsize=11,
        ylims=(0.0, 1.0), xlims=(1.0, 4.0),
        background_color=:transparent, foreground_color=:white,
        bottom_margin=8Plots.mm, left_margin=6Plots.mm)

    for (k_c, lbl, col) in [
            (Inf, "k → ∞   (Poisson)",          :steelblue),
            (1.0, "k = 1    (SIR / geometric)", :goldenrod),
            (0.1, "k = 0.1  (superspreading)",  :crimson)]
        plot!(fig, R₀_range, final_size.(R₀_range, k_c); color=col, label=lbl)
    end

    vline!(fig, [R₀_g]; color=:gray, lw=1.2, linestyle=:dot, label="")
    annotate!(fig, R₀_g + 0.05, 0.06, text("school  (R₀ = $R₀_g)", :gray, :left, 9))

    for (k_c, col) in [(Inf, :steelblue), (1.0, :goldenrod), (0.1, :crimson)]
        scatter!(fig, [R₀_g], [final_size(R₀_g, k_c)]; color=col, ms=5, label="")
    end

    fig
end

# ╔═╡ c0d1e2f3-0301-11ef-0301-1a2b3c4d5e6f
md"""
## III. Networks and Bottlenecks

Branching processes tell us *when* an outbreak takes off, and superspreading tells us that a few individuals drive it — but both assume **homogeneous mixing**: every pair of people has the same contact probability. Real populations have structure: households, schools, workplaces, travel corridors.

Modeling an epidemic as a **contact network** (nodes = people, edges = potential transmission routes) lets us apply the mathematics of percolation theory to ask whether infection turns into an outbreak based on community structure.

Two important caveats this model still carries:
- **Static contacts** — the network is fixed; it ignores social distancing once people become aware.
- **Uniform edge transmissibility** — every contact has the same transmission probability, ignoring that living with someone is very different from passing them in public.
"""

# ╔═╡ c0d1e2f3-0302-11ef-0302-1a2b3c4d5e6f
md"""
### Bond Percolation and Transmissibility

Each edge is **independently retained** with probability $T$ — the *transmissibility*, defined as the probability that infection crosses a contact given exposure. For the SIR model this is

$$T = \frac{\beta}{\beta + \gamma}$$

the probability that transmission occurs before the infectious period ends. The infected cluster reachable from a seed (the index case) is the connected component of the percolated subgraph.

**Outbreak threshold (Molloy–Reed criterion):** a giant outbreak occurs when

$$T > T_c = \frac{\langle k \rangle}{\langle k^2 \rangle - \langle k \rangle}$$

where $k$ is node degree (number of contacts). For an **Erdős–Rényi** graph the degree distribution is Poisson, giving $\langle k^2 \rangle - \langle k \rangle = \langle k \rangle^2$ and $T_c = 1/\langle k \rangle$. For a **Barabási–Albert** scale-free graph ($P(k) \sim k^{-3}$), the hubs inflate $\langle k^2 \rangle$ so much that $T_c \approx 0$ — even a very weakly transmissible disease is very likely to find a pathway through the hubs and spread widely.

"""

# ╔═╡ c0d1e2f3-0303-11ef-0303-1a2b3c4d5e6f
let
    # ── Parameters ─────────────────────────────────────────────────────────
    N      = 2_000     # nodes per graph
    avg_k  = 6         # target average degree
    n_T    = 50        # number of T values swept
    n_rep  = 200       # Monte Carlo reps per T value

    # ── Bond percolation: fraction in giant component ─────────────
    function percolate_frac(g, T)
        n      = nv(g)
        active = [Int[] for _ in 1:n]
        for e in edges(g)
            if rand() < T
                push!(active[src(e)], dst(e))
                push!(active[dst(e)], src(e))
            end
        end
        visited = falses(n)
        max_sz  = 0
        for s in 1:n
            visited[s] && continue
            queue = Int[s]; visited[s] = true; sz = 1; head = 1
            while head ≤ length(queue)
                u = queue[head]; head += 1
                for nb in active[u]
                    visited[nb] && continue
                    visited[nb] = true; push!(queue, nb); sz += 1
                end
            end
            max_sz = max(max_sz, sz)
        end
        max_sz / n
    end

    # ── Build the two graphs using Graphs.jl ─────────────────────
    g_er = erdos_renyi(N, avg_k / (N - 1))
    g_ba = barabasi_albert(N, 3)       # m = 3  ⟹  avg degree ≈ 2m = 6

    d_er  = degree(g_er);  k1_er = sum(d_er)/N;  k2_er = sum(d_er .^ 2)/N
    d_ba  = degree(g_ba);  k1_ba = sum(d_ba)/N;  k2_ba = sum(d_ba .^ 2)/N

    Tc_er = k1_er / (k2_er - k1_er)
    Tc_ba = k1_ba / (k2_ba - k1_ba)

    # ── Sweep transmissibility ────────────────────────────────────────────
    T_vals  = range(0.0, 1.0, length=n_T)
    frac_er = [sum(percolate_frac(g_er, T) for _ in 1:n_rep) / n_rep for T in T_vals]
    frac_ba = [sum(percolate_frac(g_ba, T) for _ in 1:n_rep) / n_rep for T in T_vals]

    # ── Plot ─────────────────────────────────────────────────────
    fig = plot(T_vals, frac_er;
               color=:steelblue, lw=2.5,
               label="Erdős–Rényi  (⟨k⟩ ≈ $(round(k1_er, digits=1)))",
               xlabel="Transmissibility  T",
               ylabel="Fraction infected  z",
               title="Bond percolation on ER vs. scale-free graph  (N = $N,  ⟨k⟩ ≈ $avg_k)",
               titlefontsize=12, guidefontsize=12, tickfontsize=10,
               size=(820, 480), legend=:topleft, ylims=(0, 1),
               background_color=:transparent, foreground_color=:white,
               bottom_margin=8Plots.mm, left_margin=6Plots.mm)

    plot!(fig, T_vals, frac_ba;
          color=:crimson, lw=2.5,
          label="Barabási–Albert  (⟨k⟩ ≈ $(round(k1_ba, digits=1)))")

    vline!(fig, [Tc_er]; color=:steelblue, lw=1.5, linestyle=:dash,
           label="Tᶜ (ER) ≈ $(round(Tc_er, digits=3))")
    vline!(fig, [Tc_ba]; color=:crimson, lw=1.5, linestyle=:dash,
           label="Tᶜ (BA) ≈ $(round(Tc_ba, digits=3))")

    fig
end

# ╔═╡ c0d1e2f3-0304-11ef-0304-1a2b3c4d5e6f
md"""
### 2. Bottlenecks: Conductance and λ₂

For a set of nodes $S$, the **conductance** is

$$\Phi(S) = \frac{|\partial S|}{\operatorname{vol}(S)}, \qquad \operatorname{vol}(S) = \sum_{i\in S} k_i$$

where $|\partial S|$ counts edges leaving $S$. The **Cheeger constant** $h(G) = \min_S \Phi(S)$ identifies the worst bottleneck. The **spectral gap** $\lambda_2$ (second-smallest eigenvalue of the normalized Laplacian $L = I - D^{-1/2}AD^{-1/2}$) satisfies the **Cheeger inequality**

$$\frac{\lambda_2}{2} \;\le\; h(G) \;\le\; \sqrt{2\lambda_2}$$

Large $\lambda_2$ means fast mixing — infection spreads everywhere or nowhere; small $\lambda_2$ signals a narrow bridge that few edge deletions can sever.
"""

# ╔═╡ c0d1e2f3-0305-11ef-0305-1a2b3c4d5e6f
# Normalized-Laplacian spectral gap  λ₂(G)
function graph_lambda2(g)
    N   = nv(g)
    N < 2 && return 0.0
    deg = degree(g)
    all(==(0), deg) && return 0.0
    A   = Float64.(Array(adjacency_matrix(g)))
    d   = [di > 0 ? 1/sqrt(Float64(di)) : 0.0 for di in deg]
    L   = Matrix{Float64}(I, N, N) - Diagonal(d) * A * Diagonal(d)
    eigvals(Symmetric(L))[2]
end

# ╔═╡ c0d1e2f3-0306-11ef-0306-1a2b3c4d5e6f
md"""
### 3. Containment Result

**Result.** Let $S$ be a community and $\partial S$ the edges crossing into the rest of the population. Since $|\partial S| = \Phi(S)\cdot\operatorname{vol}(S)$, the probability that infection fails to cross the cut is

$$P(\text{no transmission across cut}) = (1-T)^{|\partial S|} = (1-T)^{\Phi(S)\cdot\operatorname{vol}(S)}$$

*Proof.* Each edge in $\partial S$ is independently active with probability $T$, so the probability that **all** crossing edges are inactive is $(1-T)^{|\partial S|}$. $\square$

The best-protected community is the one achieving the **Cheeger cut** $h(G) = \min_S \Phi(S)$: minimizing conductance minimizes $|\partial S|$ relative to community size, maximizing $(1-T)^{|\partial S|}$ and making containment most likely.

**Eyam, 1666.** The Derbyshire village of Eyam enforced self-quarantine during the plague — the vicar persuaded residents to remain inside a cordon. In graph terms, Eyam was the already-infected set $S$, and by cutting off nearly all contact with surrounding villages they drove $|\partial S| \to 0$, making $(1-T)^{|\partial S|} \approx 1$. Eyam was, in effect, the Cheeger set: the community that minimized conductance to the outside, giving neighboring villages maximum protection. The outbreak burned through Eyam but stopped at its boundary.
"""

# ╔═╡ c0d1e2f3-0307-11ef-0307-1a2b3c4d5e6f
md"""
### 4. Stochastic Block Model Demo

Two communities of $n_1 = n_2 = 150$ nodes, within-community edge probability $p_\text{in} = 0.08$ (average internal degree ≈ 12). We vary the number of **crossing edges** $|\partial S| \in \{0,3,\ldots,30\}$ and ask: given an outbreak seeded in community 1, does it invade community 2? The dashed curve $1-(1-T)^{|\partial S|}$ is the theoretical upper bound on $P(\text{invasion})$ from the containment result.
"""

# ╔═╡ c0d1e2f3-0308-11ef-0308-1a2b3c4d5e6f
@bind T_sbm Slider(0.1:0.05:0.5; default=0.3, show_value=true)

# ╔═╡ c0d1e2f3-0309-11ef-0309-1a2b3c4d5e6f
let
    n1, n2   = 150, 150
    p_in     = 0.08
    cuts     = 0:3:30        # crossing-edge counts (0, 3, 6, …, 30)
    n_trial  = 200

    # Build SBM: n1+n2 nodes, within-community ER edges + n_cross crossing edges
    function build_sbm(n1, n2, p_in, n_cross)
        N = n1 + n2
        g = SimpleGraph(N)
        for i in 1:n1,   j in i+1:n1;   rand() < p_in && add_edge!(g, i, j); end
        for i in n1+1:N, j in i+1:N;    rand() < p_in && add_edge!(g, i, j); end
        added = 0
        while added < n_cross
            u = rand(1:n1); v = rand(n1+1:N)
            !has_edge(g, u, v) && (add_edge!(g, u, v); added += 1)
        end
        g
    end

    # Does an outbreak seeded in community 1 invade community 2?
    function invaded(g, T, n1)
        N = nv(g)
        act = [Int[] for _ in 1:N]
        for e in edges(g)
            rand() < T || continue
            push!(act[src(e)], dst(e)); push!(act[dst(e)], src(e))
        end
        seed = rand(1:n1)
        vis  = falses(N); vis[seed] = true
        queue = Int[seed]; head = 1
        while head ≤ length(queue)
            u = queue[head]; head += 1
            for nb in act[u]
                vis[nb] && continue
                vis[nb] = true; push!(queue, nb)
            end
        end
        any(vis[n1+1:N])
    end

    # Conductance of the S = community-1 cut
    function conductance(g, n1)
        deg = degree(g)
        cut = count(e -> (src(e) ≤ n1) ⊻ (dst(e) ≤ n1), edges(g))
        vol = min(sum(deg[1:n1]), sum(deg[n1+1:nv(g)]))
        vol == 0 ? 0.0 : cut / vol
    end

    cut_vals = collect(cuts)
    nc       = length(cut_vals)
    P_inv    = zeros(nc)
    lam2     = zeros(nc)
    phi      = zeros(nc)

    for (i, ncross) in enumerate(cut_vals)
        g         = build_sbm(n1, n2, p_in, ncross)
        P_inv[i]  = sum(invaded(g, T_sbm, n1) for _ in 1:n_trial) / n_trial
        lam2[i]   = graph_lambda2(g)
        phi[i]    = conductance(g, n1)
    end

    theory = [1 - (1 - T_sbm)^ncross for ncross in cut_vals]

    p1 = plot(cut_vals, P_inv;
              color=:steelblue, lw=2.5, label="Simulated P(invasion)",
              xlabel="|\u2202S|  (crossing edges)", ylabel="P(community 2 invaded)",
              title="Invasion probability  (T = $T_sbm)",
              titlefontsize=12, guidefontsize=11, tickfontsize=10,
              ylims=(0, 1), legend=:bottomright,
              background_color=:transparent, foreground_color=:white,
              bottom_margin=8Plots.mm, left_margin=6Plots.mm)
    plot!(p1, cut_vals, theory;
          color=:gray70, lw=1.5, linestyle=:dash,
          label="1−(1−T)^|\u2202S|  (upper bound)")

    p2 = plot(cut_vals, lam2;
              color=:darkorange, lw=2.5, label="λ₂",
              xlabel="|\u2202S|  (crossing edges)", ylabel="Value",
              title="Spectral gap λ₂ and conductance Φ",
              titlefontsize=12, guidefontsize=11, tickfontsize=10,
              legend=:topleft,
              background_color=:transparent, foreground_color=:white,
              bottom_margin=8Plots.mm, left_margin=6Plots.mm)
    plot!(p2, cut_vals, phi;
          color=:orchid, lw=2.5, linestyle=:dot, label="Φ(S)")

    plot(p1, p2; layout=(1,2), size=(1000, 420))
end

# ╔═╡ c0d1e2f3-0400-11ef-0400-1a2b3c4d5e6f
md"""
## Summary

Each framework captures a different layer of epidemic complexity, with its own assumptions:

| Model | Key output | Assumes |
|---|---|---|
| **Deterministic SIR** | Expected trajectories $S(t), I(t), R(t)$; $R_0$; final size $z$ | Large homogeneous population; no randomness |
| **Stochastic (CTMC)** | Outbreak size distribution; extinction probability; fade-out | Homogeneous mixing; finite $N$ |
| **Branching process** | Fizzle probability $q^*$; superspreading via dispersion $k$ | Independent offspring; valid only in early phase before susceptible depletion |
| **Network percolation** | Threshold $T_c = \langle k\rangle/(\langle k^2\rangle - \langle k\rangle)$; invasion probability; $\lambda_2$, $\Phi$ | Static contacts; uniform edge transmissibility; ignores blocking by recovered nodes |

**Other directions not covered here:**
- *Spatial SIR* (Kendall, 1965): reaction-diffusion PDE on a continuous landscape; produces traveling infection wavefronts at speed $v \propto \sqrt{R_0 - 1}$.
- *Metapopulation models*: patch-coupled SIR systems (cities linked by commuting or air travel); used for pandemic forecasting.
- *Agent-based models*: simulate each individual with heterogeneous behavior; most realistic but least analytically tractable.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
CSV = "~1.1.0"
DataFrames = "~1.8.2"
Distributions = "~0.25.132"
Graphs = "~1.13.1"
Plots = "~1.41.7"
PlutoUI = "~0.7.84"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.1"
manifest_format = "2.1"
project_hash = "6ecc818fc7d02eebf1a72d41af14c16555570f24"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "8c0bc12397dabf5d2dfc4b5763ec463854cc44d3"
registries = "General"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.46"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
registries = "General"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "DataStrings", "Dates", "Downloads", "Durations", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "Printf", "Tables", "Unicode"]
git-tree-sha1 = "ed05d41d96b1a815d52e9de1ef472971bc509832"
registries = "General"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "1.1.0"

    [deps.CSV.extensions]
    CSVDataDecimalsExt = "DataDecimals"
    CSVFilePathsBaseExt = "FilePathsBase"
    CSVInlineStringsExt = "InlineStrings"

    [deps.CSV.weakdeps]
    DataDecimals = "3e2245cb-6932-498d-a7cf-39d39de8bde3"
    FilePathsBase = "48062228-2e41-5def-b9a4-89aafe57970f"
    InlineStrings = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "7b841680738c19948120f6e4cf8d0200518bd564"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.8+0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "970758a3d591a2a5c2a907c53f2e2f8c1b1d3537"
registries = "General"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.9"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "a7cf827a79b378a381e31e5c7feca853ec19cb94"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.4"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "a633156e74096f90b3b5e833529292b651b5671a"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.3"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
registries = "General"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
registries = "General"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
registries = "General"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
registries = "General"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStrings]]
git-tree-sha1 = "b8032263d364177a0b192ea15f277df4e6ab07e2"
registries = "General"
uuid = "48204bd6-5611-42ea-b167-fc1d713f9be2"
version = "1.1.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
registries = "General"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
registries = "General"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
registries = "General"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "2ddbc341251ed713d39e503b504593a5d6789551"
registries = "General"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.132"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "38ef52f18cbdfa783ec6c25acc23fb1e8525ec16"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.10.2"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.Durations]]
deps = ["Dates"]
git-tree-sha1 = "407b209d76168dcc4355f9dcc4d3b3aaa60dfb3e"
registries = "General"
uuid = "41ac09f5-0a95-4810-aa18-c88bf48ee130"
version = "1.4.1"

    [deps.Durations.extensions]
    DurationsArrowExt = "Arrow"
    DurationsTimeZonesExt = "TimeZones"

    [deps.Durations.weakdeps]
    Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
    TimeZones = "f269a46b-ccf7-5d73-abea-4c690281aa53"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
registries = "General"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2bfb1e047e2ad0a5ca94365340bde8005d637568"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "7bd13840b4148949e290748071bbe6826b3eabcd"
registries = "General"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d9d3cd382f2c999f684e6bfa4039fcc16c153b6f"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "9.0.2+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "831f86fde78ab7f6470d6ba4d1161a11fb6b4ec0"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.18.2"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "64bbbb7d1499297751b536dd39c58b20750ab1db"
registries = "General"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.5.1+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "4d777f73c46b46b8b5276206059cf8a195499314"
registries = "General"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.27"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f8eb8f7ba13ea75083531647fc8faeda8d541f07"
registries = "General"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.27+0"

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "becc397f7cfb06e343496ae6ffb04818a851da51"
registries = "General"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
registries = "General"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "ea9f1fb37ff10597353b56f3ba0ebd525acef6e0"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.90.1+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7a98c6502f4632dbe9fb1973a4244eaa3324e84d"
registries = "General"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "9d9531a9cb63a9edc33836414e82a07e81710de2"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.14004.0+0"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "c28dceeadc1c9c699656a3ac9d63f79cfaa17472"
registries = "General"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.31"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
registries = "General"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InlineStrings]]
git-tree-sha1 = "06b65886c7577a3784d616e29f1302c2e36e389d"
registries = "General"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.6"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "5cc6ad3e2bfc75a11eed92632ee341d37f4a4536"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.18"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
registries = "General"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "d22e8bdb3f03c300a103ff23c0dd5a654c6701de"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.7"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
registries = "General"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
registries = "General"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "9939eb6f799a963b3392ac4a1d49d3fc0646eb80"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.11.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "39bca05343661c347aae0bca57a5994a0bf4f08d"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.2.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
registries = "General"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.18.0+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.1+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.104+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
registries = "General"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b8a91037379e1fd661e9b5ff3d7a770424b1f9f2"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9e887184f306b28dbfcff88cefbb0ec80f1aa9e0"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.4+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "28d620bcdb37a3fb02de9f521928ee453690f414"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.3"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
registries = "General"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "d9d20d6eb2a3290ba5c70a1c013b45305cc39c9e"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.8.13"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "b079a6affebde167dad8ffe0d670b53917fbcb68"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.46.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
registries = "General"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
registries = "General"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
registries = "General"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Reexport", "Statistics"]
git-tree-sha1 = "ddcc36fd83fb4cd459a53c891981697bde0d3de6"
registries = "General"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.5.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "83bd514e8ff16b5858ac54c53fa0bcf6002a3b00"
registries = "General"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.7"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "9869376035f47756b8b8e711fd0d06a06d51da6f"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.84"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
registries = "General"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "aef1987d90dcc2e2cbd919ad10b2d25bcce4f418"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "StyledStrings", "Tables"]
git-tree-sha1 = "99fd8f32ab6728ff205cddd24e2d9ea6cbb498d3"
registries = "General"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.5.0"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"
    PrettyTablesWriteDocxExt = "WriteDocx"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    WriteDocx = "d049ceea-54ee-41d7-a26f-ba29db3b6599"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
registries = "General"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
registries = "General"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
registries = "General"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
registries = "General"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
registries = "General"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
registries = "General"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "0c0dcffc464cda2ff33fc8ffc1dd1bdd53cd98be"
registries = "General"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.4.0"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
registries = "General"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
registries = "General"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
registries = "General"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "c5e8a0a52c0d5d043214de04f11f4d9c2c6ccdb8"
registries = "General"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.13"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
registries = "General"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates"]
git-tree-sha1 = "8238217340ad0aaabe11afe39c1098b5bc9f4c8e"
registries = "General"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.1.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
registries = "General"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "88ef0d98204515bc53dd4aaea5f7788882d1b66c"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "39e70e0ab5d7f89833a62ab7c79df15d4fc417c1"
registries = "General"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.22"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
registries = "General"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91a5737baed20ee31f3faea0e51f57461f6a689e"
registries = "General"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools", "StyledStrings"]
git-tree-sha1 = "9e3a02d73a8f0a9be04e677dbef758aab40f8bea"
registries = "General"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.6.1"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ad435e30854ed0f50f4b7f67edd71664afecc5d8"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.10.0"

    [deps.StructUtils.extensions]
    StructUtilsDataFramesExt = ["DataFrames"]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazilyInitializedFields = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
registries = "General"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
registries = "General"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
registries = "General"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
deps = ["Serialization"]
git-tree-sha1 = "eb3b85ca904a1d1647b15acc091b113b8874b648"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
registries = "General"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
registries = "General"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
registries = "General"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
registries = "General"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
registries = "General"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "ce91b54a5f90c4f27df6950706234b34e49cf79e"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.14+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
registries = "General"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "704274ca960c42049fd9bda089fc4be32af5cdca"
registries = "General"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.5+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
registries = "General"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
registries = "General"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a54cf54677c9f6eac0d97e9d757c233b56bc21b9"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.13+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
registries = "General"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
registries = "General"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
registries = "General"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
registries = "General"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
registries = "General"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
registries = "General"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
registries = "General"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "9a5e6e8f1a7c40941465c04840145e527f0d6b4b"
registries = "General"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.5.0+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "df77b22eb77051b3d0ef30aab979684f7482c66f"
registries = "General"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.48.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
registries = "General"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
registries = "General"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1210ba774d3427387d307bf1f416d699b7c39417"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.15.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
registries = "General"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
registries = "General"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
registries = "General"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "1fcd6cf05d23adab0b5ffc496d020537215751dc"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.59+1"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
registries = "General"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.67.1+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.2+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
registries = "General"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╠═c3a1f2a0-9e01-11ef-0001-1a2b3c4d5e6f
# ╠═c3a1f2a0-9e01-11ef-0002-1a2b3c4d5e6f
# ╠═c3a1f2a0-9e01-11ef-0003-1a2b3c4d5e6f
# ╟─c3a1f2a0-9e01-11ef-0005-1a2b3c4d5e6f
# ╟─c3a1f2a0-9e01-11ef-0004-1a2b3c4d5e6f
# ╠═d4b2e3c0-ae02-11ef-0006-1a2b3c4d5e6f
# ╠═d4b2e3c0-ae02-11ef-0008-1a2b3c4d5e6f
# ╠═d4b2e3c0-ae02-11ef-0009-1a2b3c4d5e6f
# ╟─d4b2e3c0-ae02-11ef-0007-1a2b3c4d5e6f
# ╠═f6d4a5e0-c004-11ef-0011-1a2b3c4d5e6f
# ╠═e1f2a3b4-0028-11ef-0028-1a2b3c4d5e6f
# ╠═e1f2a3b4-0029-11ef-0029-1a2b3c4d5e6f
# ╠═e1f2a3b4-0030-11ef-0030-1a2b3c4d5e6f
# ╠═e1f2a3b4-0031-11ef-0031-1a2b3c4d5e6f
# ╠═e5c3f4d0-bf03-11ef-0010-1a2b3c4d5e6f
# ╠═a0b1c2d3-0410-11ef-0410-1a2b3c4d5e6f
# ╠═a0b1c2d3-0412-11ef-0412-1a2b3c4d5e6f
# ╠═a0b1c2d3-0413-11ef-0413-1a2b3c4d5e6f
# ╠═a0b1c2d3-0414-11ef-0414-1a2b3c4d5e6f
# ╟─a7e5d2c0-d105-11ef-0012-1a2b3c4d5e6f
# ╠═a7e5d2c0-d105-11ef-0013-1a2b3c4d5e6f
# ╟─b8f6e3d0-e206-11ef-0014-1a2b3c4d5e6f
# ╠═e1f2a3b4-0032-11ef-0032-1a2b3c4d5e6f
# ╠═e1f2a3b4-0033-11ef-0033-1a2b3c4d5e6f
# ╠═e1f2a3b4-0034-11ef-0034-1a2b3c4d5e6f
# ╟─e1f2a3b4-0035-11ef-0035-1a2b3c4d5e6f
# ╟─b8f6e3d0-e206-11ef-0015-1a2b3c4d5e6f
# ╟─c9e7b4f0-f307-11ef-0016-1a2b3c4d5e6f
# ╠═d1e2f3a4-0016-11ef-0016-1a2b3c4d5e6f
# ╠═d1e2f3a4-0017-11ef-0017-1a2b3c4d5e6f
# ╟─d1e2f3a4-0017-11ef-00fa-1a2b3c4d5e6f
# ╠═d1e2f3a4-0017-11ef-00fb-1a2b3c4d5e6f
# ╠═d1e2f3a4-0018-11ef-0018-1a2b3c4d5e6f
# ╟─d1e2f3a4-0019-11ef-0019-1a2b3c4d5e6f
# ╟─d1e2f3a4-0020-11ef-0020-1a2b3c4d5e6f
# ╠═d1e2f3a4-0021-11ef-0021-1a2b3c4d5e6f
# ╠═d1e2f3a4-0022-11ef-0022-1a2b3c4d5e6f
# ╠═d1e2f3a4-0024-11ef-0024-1a2b3c4d5e6f
# ╠═d1e2f3a4-0025-11ef-0025-1a2b3c4d5e6f
# ╟─d1e2f3a4-0026-11ef-0026-1a2b3c4d5e6f
# ╠═c0d1e2f3-0300-11ef-0300-1a2b3c4d5e6f
# ╟─c0d1e2f3-0301-11ef-0301-1a2b3c4d5e6f
# ╠═c0d1e2f3-0302-11ef-0302-1a2b3c4d5e6f
# ╟─c0d1e2f3-0303-11ef-0303-1a2b3c4d5e6f
# ╠═c0d1e2f3-0304-11ef-0304-1a2b3c4d5e6f
# ╠═c0d1e2f3-0305-11ef-0305-1a2b3c4d5e6f
# ╠═c0d1e2f3-0306-11ef-0306-1a2b3c4d5e6f
# ╠═c0d1e2f3-0307-11ef-0307-1a2b3c4d5e6f
# ╠═c0d1e2f3-0308-11ef-0308-1a2b3c4d5e6f
# ╟─c0d1e2f3-0309-11ef-0309-1a2b3c4d5e6f
# ╠═c0d1e2f3-0400-11ef-0400-1a2b3c4d5e6f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
