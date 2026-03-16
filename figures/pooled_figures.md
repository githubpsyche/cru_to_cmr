# Figure 1

## Caption
Summary statistic fits of the baseline CMR model to PEERS data.
**Left**: probability of recall initiation by serial position.
**Middle**: conditional response probability as a function of lag.
**Right**: recall probability by serial position.

## Alt Text
Three side-by-side plots comparing human free-recall data to the CMR model. Left: Recall-initiation probabilities by study position show a strong recency peak (last item) and a smaller primacy peak (first item); model follows the same U-shape. Middle: Lag-CRP curve is forward-skewed, with the highest transition at +1; model tracks this asymmetry. Right: Overall recall accuracy by study position forms a shallow U across the 16 positions; model slightly underestimates primacy and final-item recall but captures the general trend.

# Figure 2

## Caption
Summary statistic fits of baseline CRU (**Top**), CRU with free start context integration rate $\beta_\text{start}$ (**Middle**), and CRU freeing both start context integration rate ($\beta_\text{start}$) and primacy gradient ($\phi_\text{s}$ and $\phi_\text{d}$) parameters (**Bottom**) to PEERS free recall data.
**Left**: probability of recall initiation by serial position.
**Middle**: conditional response probability as a function of lag.
**Right**: recall probability by serial position.

## Alt Text
A comparison of three CRU variants against empirical free-recall patterns. Rows index by model variants: Top: baseline CRU; middle: CRU with a free start-context integration rate ($\beta_\text{start}$); bottom: CRU with both $\beta_\text{start}$ and an associative primacy gradient ($\phi_\text{s}$ and $\phi_\text{d}$). Columns index summary measures. First-recall curve (left column): probability that recall begins with each serial position. A steep rise at the end marks the recency effect (late items recalled first); a smaller peak at the start marks the primacy effect (early items sometimes recalled first). Lag-CRP (center column): probability of transitioning between recalled items separated by a given study lag. Tall bars at +1 show the forward short-lag preference; smaller bars at –1 show a smaller preference for backward short-lag transitions. Serial-position curve (right column): overall recall probability for each position—high at the start (primacy) and end (recency). Take-away: Freeing $\beta_\text{start}$ lets the model capture the strong recency start, while adding the primacy gradient boosts early-item recall and heightens ±1 lag peaks, bringing all three panels much closer to the data.

# Figure 3

## Caption
Simulation of the impact of shifting the start context integration rate parameter $\beta_\text{start}$ on the probability of starting recall by serial position (**Left**) and the recall probability by serial position (**Right**) for CMR.
Using parameters fit to PEERS free recall data, $\beta_\text{start}$ is shifted from 0 to 1 in increments of 0.1, with the color of the lines indicating the value of the parameter.


## Alt Text
Two side-by-side line charts (using a shared color legend with values 0.0 -- 0.9) illustrate how gradually increasing the start-of-list context-integration parameter $\beta_\text{start}$ reshapes simulated recall behavior in CMR. Left panel -- "Probability of N-th Recall": For 16 study  positions on the x-axis, each colored line shows the chance that the first item recalled (i.e., recall initiation) comes from that position. Lower $\beta_\text{start}$ values (light gray) yield a steep recency peak at the final position, whereas higher values (dark gray/black) progressively shift initiation toward the first item, producing a strong primacy peak. Right panel -- "Recall Rate": The same color-coded lines plot overall recall probability for every serial position. As $\beta_\text{start}$ increases, the recency advantage at the end of the list diminishes while recall of early positions improves, creating the classic U-shaped primacy--recency curve. Together the panels show that $\beta_\text{start}$ governs how strongly the end-of-list context is blended back toward the start: small values favor recency-driven cueing, large values favor primacy-driven cueing, and intermediate values balance the two.

# Figure 4

## Caption
Summary statistic fits of models to the PEERS free recall dataset [@healey2014memory].
**Top**: CRU with free pre-experimental context-to-feature memory ($\alpha$, $\delta$), primacy gradient ($\phi_\text{s}$, $\phi_\text{d}$), and start context integration rate ($\beta_\text{start}$) parameters.
**Middle**: CRU with free item-to-context learning rate ($\gamma$), primacy gradient ($\phi_\text{s}$, $\phi_\text{d}$), and start context integration rate ($\beta_\text{start}$) parameters.
**Bottom**: CRU with free item-to-context learning rate ($\gamma$), pre-experimental context-to-feature memory ($\alpha$, $\delta$), primacy gradient ($\phi_\text{s}$, $\phi_\text{d}$), and start context integration rate ($\beta_\text{start}$) parameters -- equivalent to CMR.
**Left**: Probability of starting recall by serial position.
**Left**: Probability of starting recall by serial position.
**Middle**: Conditional response probability as a function of lag.
**Right**: Recall probability by serial position.

## Alt Text
Nine mini-plots arranged in a 3 × 3 grid compare *model* (black) to *human data* (gray) for three increasingly complex CRU/CMR variants. Rows (top -> bottom) show, respectively: (1) CRU + pre-experimental support + primacy gradient, (2) CRU + feature-to-context learning + primacy gradient, (3) Full CMR (all mechanisms enabled). Columns (left -> right) display three benchmark statistics for 16-word free-recall lists. Left column: Recall-initiation curve. Y: probability the first recall comes from each study position. High right-end values illustrate the recency effect (participants often start with the last-studied word), whereas smaller left-end bumps reflect primacy (some start with the first word). Middle column: Lag-conditional response probability (lag-CRP). X: positional lag between successive recalls; Y: conditional probability. The sharp forward spike at +1 and the gentler backward spike at -1 indicate a short-lag contiguity bias; people tend to move to temporally adjacent items, more so forward than backward. Right column: Serial-position curve (SPC). Y: overall recall rate for each study position. The U-shape reprises primacy (higher accuracy for early items) and recency (late-item advantage after a dip in the middle). Error bars show ±1 SE. Progressing down the rows shows that adding each CMR mechanism successively narrows the gap between model curves and gray data points: the forward and backward peaks in the lag-CRP grow taller, and the SPC's early-item accuracy rises, demonstrating better fits to primacy, recency, and short-lag phenomena.

# Figure 5

## Caption
Simulation of the impact of shifting CMR's $\gamma$ (**Left**) and $\delta$ (**Right**) parameters on the conditional response probability as a function of lag for CMR.
Using parameters fit to @healey2014memory, the learning rate parameter $\gamma$ is shifted from 0 to 1 in increments of 0.1, and the item support parameter $\delta$ is shifted from 0 to 10 in increments of 1, with the color of the lines indicating the value of the parameter.

## Alt Text
Two side-by-side line plots show how changing two CMR parameters alters the lag-conditional response probability (lag-CRP). The left plot varies the learning-rate parameter $\gamma$ from 0.0 (light gray) to 0.9 (dark gray / black) in 0.1 steps: larger $\gamma$ sharply boosts the probability of a +1 backward transition (lag –1) while slightly reducing the –3 to –5 lags. The right plot varies the self-support parameter $\delta$ from 0.0 (light gray) to 8.9 (dark gray / black) in unit steps: higher $\delta$ steepens both the forward +1 and backward –1 peaks while depressing longer-lag transitions. Each coloured line therefore traces how strengthening either parameter concentrates recall transitions around neighbouring items, with the legend listing the parameter values.

# Figure 6

## Caption
Summary statistic fits of CMR with CRU's context-based recall termination mechanism to @healey2014memory.
**Left**: probability of starting recall by serial position.
**Middle**: conditional response probability as a function of lag.
**Right**: recall probability by serial position.

## Alt Text
Three side-by-side line plots compare a CMR model that uses CRU's context-based stopping rule (black) with empirical free-recall data from @healey2014memory (gray). Left panel: probability that each study position is produced as the N-th recall; both model and data rise steeply at the end of the list, though the model sits slightly higher at the final positions. Middle panel: lag-conditional response probability, showing a forward-skewed peak at +1; the model captures the overall shape but undershoots the sharpness of the +1 jump. Right panel: overall recall rate by study position; both traces form a shallow "U", but the model overestimates mid-list recall and underestimates the depth of the primacy dip.

# Figure 7

## Caption
Serial recall accuracy (SRAC) fits to @logan2021serial serial recall data for list lengths of 5 (**Left Column**), 6 (**Middle Column**), 7 (**Right Column**) of baseline CRU (**Top**), the best performing CRU variant with free pre-experimental context-to-feature memory ($\alpha$, $\delta$) and CMR-specific primacy gradient  ($\phi_\text{s}$, $\phi_\text{d}$) parameters (**Middle**), and CMR with its default position-based recall termination mechanism and CRU's item identification confusability mechanism (**Bottom**).

## Alt Text
Nine‐panel line chart comparing observed versus model-predicted *serial recall accuracy* (SRAC). Columns correspond to list lengths 5 (left), 6 (middle), and 7 (right). Rows show, top to bottom: (1) baseline CRU; (2) best-performing CRU variant that adds CMR’s associative primacy gradient plus pre-experimental context–feature support; (3) CMR with its native position-based stop rule but augmented with CRU’s item-confusability mechanism. In every panel, study position (x-axis) runs from first to last list item; SRAC (y-axis) runs from 0.3–1.0. Greay lines with error bars plot human data; black lines plot model fits. All models capture the overall primacy gradient (higher accuracy for early study positions) but diverge in later positions: baseline CRU underestimates late-list accuracy, the hybrid CRU variant closes that gap, and the CMR-based model slightly overestimates mid-list accuracy for longer lists.

# Figure 8

## Caption
Intrusion, order, and omission error rates (top, middle, and bottom rows respectively) by serial position for list lengths 5, 6, and 7 (left, center, and right columns), in @logan2021serial serial recall data.
Lines compare observed error rates with predicted error rates from best performing CRU variant with free pre-experimental context-to-feature memory ($\alpha$, $\delta$) and CMR-specific primacy gradient  ($\phi_\text{s}$, $\phi_\text{d}$) parameters.


## Alt Text
Nine mini-plots arranged in a 3 by 3 grid. Rows represent error type during serial recall of letters. Top row – Intrusion errors: recalling a letter that was not on the study list. Middle row – Order errors: recalling a studied letter but in the wrong serial position. Bottom row – Omission errors: failing to supply any letter for a position. Columns represent list length: 5-item lists (left), 6-item lists (centre), 7-item lists (right). Within each panel, grey points/lines plot observed error rates by study position; black points/lines plot predicted rates from the best-fitting CRU + CMR hybrid model (which adds pre-experimental context-to-item associations and a primacy gradient). Error bars show ±1 SE. The model tracks the upward error trend across later positions and captures the different magnitudes for intrusion, order and omission errors across list lengths.

# Figure 9

## Caption
Lag-conditional response probability (lag-CRP) fits of baseline CRU (**Top**); best performing CRU variant with free pre-experimental context-to-feature memory ($\alpha$, $\delta$) and CMR-specific primacy gradient  ($\phi_\text{s}$, $\phi_\text{d}$) parameters (**Middle**); and CMR with its default position-based recall termination mechanism and CRU's item identification confusability mechanism (**Bottom**) to @logan2021serial serial recall data.
Lines compare observed lag-CRP with predicted lag-CRP for the applicable model variant.


## Alt Text
Lag-conditional response probability (lag-CRP) curves for serial-recall lists of length 5, 6, and 7 (columns). Within each column, three rows show: (1) baseline CRU; (2) the hybrid CRU that adds primacy and pre-experimental support; (3) CMR with position-based stopping. Observed data (grey lines ± SE) and model predictions (black lines) are plotted for lags −4 to +5. All variants fit the dominant +1 forward transition, but only the hybrid CRU (row 2) closely tracks the small yet reliable −1 "fill-in" backward transition, while baseline CRU underestimates and CMR overestimates backward-lag probabilities beyond −1.

