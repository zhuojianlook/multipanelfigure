/* ──────────────────────────────────────────────────────────
   IntensityPlotOptionsDialog — the options behind the "Plot
   intensities" R node (double-click the node, or its Plot options
   button). Left: which graphs are drawn. Right: the settings of the
   selected graph; settings that do not apply to its type are hidden,
   settings that do nothing given the others are disabled with the
   reason. Edits stay in a draft until Apply; the node's R code is then
   regenerated from the result (utils/intensityPlotOptions).
   ────────────────────────────────────────────────────────── */

import { useEffect, useId, useMemo, useState, type ReactNode } from "react";
import {
  Alert, Box, Button, ButtonBase, Checkbox, Dialog, DialogActions, DialogContent, DialogTitle,
  FormControlLabel, IconButton, MenuItem, TextField, ToggleButton, ToggleButtonGroup, Tooltip, Typography,
  useMediaQuery, useTheme, type Theme,
} from "@mui/material";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import ArrowDownwardIcon from "@mui/icons-material/ArrowDownward";
import {
  INTENSITY_COMPARTMENT_LABELS, INTENSITY_FONT_RANGE, INTENSITY_GRAPHS, INTENSITY_GRAPH_IDS,
  INTENSITY_OPTION_CHOICES as CHOICES, INTENSITY_PANEL_RANGE, INTENSITY_X_ANGLES,
  applyIntensityGraphToAll, defaultIntensityPlotConfig, graphHasSetting, intensityGraphUnavailableReason,
  isFoldChangeGraph, normalizeIntensityPlotConfig, resetIntensityGraph, resolveGraph, resolveOrder,
  settingInactiveReason, updateIntensityGraph, yAxisStartsAtZero,
  type CompId, type IntensityGraphId, type IntensityGraphInfo, type IntensityGraphOptions,
  type IntensityGraphOverride, type IntensityPlotCodeState, type IntensityPlotConfig, type IntensityPlotContext,
  type IntensitySettingKey,
} from "../../utils/intensityPlotOptions";

export interface IntensityPlotOptionsDialogProps {
  open: boolean;
  /** The node's stored options (null = defaults). */
  initial: IntensityPlotConfig | null;
  /** Groups, channels and compartments the upstream data offers. */
  context: IntensityPlotContext;
  /** How the node's code relates to the options; "edited" shows a warning. */
  codeState: IntensityPlotCodeState;
  /** Shown in the title. */
  nodeLabel?: string;
  onClose: () => void;
  onSave: (cfg: IntensityPlotConfig) => void;
}

// ── Wording ─────────────────────────────────────────────────

type Opt = IntensityGraphOptions;
const SECTION_TITLE: Record<IntensityGraphInfo["section"], string> = {
  descriptive: "Descriptive", perImage: "Per image", foldChange: "Fold change", control: "Control",
};
const CHART_LABEL: Record<Opt["chart"], string> = { bar: "Bars", box: "Box", violin: "Violin", mean: "Mean" };
const ERROR_LABEL: Record<Opt["error"], string> = { sd: "SD", sem: "SEM", ci: "95% CI", none: "None" };
const TEST_LABEL: Record<Opt["test"], string> = {
  tukey: "Tukey HSD", welch: "Welch t-test (Holm-adjusted)", wilcox: "Wilcoxon rank-sum (Holm-adjusted)", none: "None",
};
const COMPARE_LABEL: Record<Opt["compare"], string> = { all: "All pairs", ref: "Vs reference" };
const SIG_LABEL: Record<Opt["sig"], string> = { stars: "Stars", p: "p-values" };
const LAYOUT_LABEL: Record<Opt["layout"], string> = { panels: "Side by side", dodged: "Grouped" };
const LAYOUT_HELP: Record<Opt["layout"], string> = {
  panels: "One panel per compartment (whole cell, nuclear, cytoplasm), side by side",
  dodged: "One panel per channel: compartments on the x axis, groups side by side",
};
const YSCALE_LABEL: Record<Opt["yScale"], string> = {
  shared: "Shared within each channel", independent: "Separate for every panel", same: "Same for all panels",
};
const PALETTE_LABEL: Record<Opt["palette"], string> = {
  set2: "Soft", okabe: "Colour-blind safe", dark: "Strong", grey: "Greys",
};
/** First colours of each palette (R runtime .ipo_PAL), for the swatches. */
const PALETTE_SWATCH: Record<Opt["palette"], string[]> = {
  set2: ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3"],
  okabe: ["#E69F00", "#56B4E9", "#009E73", "#F0E442"],
  dark: ["#1B9E77", "#D95F02", "#7570B3", "#E7298A"],
  grey: ["#CCCCCC", "#969696", "#636363", "#252525"],
};
const LEGEND_LABEL: Record<Opt["legend"], string> = { bottom: "Bottom", right: "Right", top: "Top", none: "None" };
const AUTO_REF = "\u0000auto";
const NOT_IN_DATA = "not in current data";

const fieldSx = {
  "& .MuiInputBase-input": { fontSize: "0.78rem" },
  "& .MuiInputLabel-root": { fontSize: "0.78rem" },
  "& .MuiFormHelperText-root": { fontSize: "0.66rem", mx: 0.25 },
} as const;
const MENU_SX = { "& .MuiMenuItem-root": { fontSize: "0.78rem", py: 0.4, minHeight: 28 } } as const;
const toggleSx = { textTransform: "none", fontSize: "0.7rem", py: 0.25, px: 1 } as const;
/** Warning-coloured small text that stays readable on dark and light paper. */
const warnColor = (t: Theme) => (t.palette.mode === "dark" ? t.palette.warning.light : t.palette.warning.dark);
const okColor = (t: Theme) => (t.palette.mode === "dark" ? t.palette.success.light : t.palette.success.dark);

// ── Small controls ──────────────────────────────────────────

function Reason({ text }: { text: string | null | undefined }) {
  if (!text) return null;
  return (
    <Typography variant="caption" sx={{ display: "block", color: "text.disabled", fontSize: "0.66rem", mt: 0.25, lineHeight: 1.3 }}>
      {text}
    </Typography>
  );
}

function ToggleSetting<T extends string | number>({ label, value, options, onChange, reason, help }: {
  label: string; value: T; options: readonly { value: T; label: string }[];
  onChange: (v: T) => void; reason?: string | null; help?: string;
}) {
  const id = useId();
  return (
    <Box sx={{ minWidth: 0 }}>
      <Typography id={id} variant="caption"
        sx={{ display: "block", mb: 0.3, fontSize: "0.7rem", color: reason ? "text.disabled" : "text.secondary" }}>
        {label}
      </Typography>
      <ToggleButtonGroup size="small" exclusive value={value} disabled={!!reason} aria-labelledby={id}
        onChange={(_, v: T | null) => { if (v !== null) onChange(v); }} sx={{ flexWrap: "wrap" }}>
        {options.map((o) => (
          <ToggleButton key={String(o.value)} value={o.value} sx={toggleSx}>{o.label}</ToggleButton>
        ))}
      </ToggleButtonGroup>
      <Reason text={reason || help} />
    </Box>
  );
}

function SelectSetting<T extends string>({ label, value, options, onChange, reason, help }: {
  label: string; value: T; options: readonly { value: T; label: ReactNode }[];
  onChange: (v: T) => void; reason?: string | null; help?: string;
}) {
  return (
    <TextField select fullWidth size="small" label={label} value={value} disabled={!!reason}
      helperText={reason || help} onChange={(e) => onChange(e.target.value as T)} sx={fieldSx}
      slotProps={{ select: { MenuProps: { slotProps: { paper: { sx: MENU_SX } } } } }}>
      {options.map((o) => <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>)}
    </TextField>
  );
}

function CheckSetting({ label, checked, onChange, reason }: {
  label: string; checked: boolean; onChange: (v: boolean) => void; reason?: string | null;
}) {
  return (
    <Box sx={{ minWidth: 0 }}>
      <FormControlLabel
        disabled={!!reason}
        control={<Checkbox size="small" checked={checked} onChange={(e) => onChange(e.target.checked)} sx={{ p: 0.5 }} />}
        label={label}
        sx={{ ml: -0.5, mr: 0, "& .MuiFormControlLabel-label": { fontSize: "0.78rem" } }}
      />
      <Reason text={reason} />
    </Box>
  );
}

/** Free text; empty means automatic. Shows what was typed, while the draft
 *  holds the trimmed value. */
function TextSetting({ label, value, onCommit }: {
  label: string; value: string | null; onCommit: (v: string) => void;
}) {
  const [text, setText] = useState(value ?? "");
  useEffect(() => {
    setText((t) => ((t.trim() || null) === value ? t : value ?? ""));
  }, [value]);
  return (
    <TextField fullWidth size="small" label={label} placeholder="Automatic" value={text} sx={fieldSx}
      onChange={(e) => { setText(e.target.value); onCommit(e.target.value); }}
      slotProps={{ inputLabel: { shrink: true }, htmlInput: { maxLength: 300 } }} />
  );
}

/** A number typed as text (a number input drops half-typed values like
 *  "2."). Values in range are taken as typed; leaving the field clamps. */
function NumberSetting({ label, value, range, step, allowEmpty, unit, onCommit }: {
  label: string; value: number | null; range: { min: number; max: number }; step: number;
  allowEmpty?: boolean; unit?: string; onCommit: (v: number | null) => void;
}) {
  const fmt = (v: number | null) => (v === null ? "" : String(v));
  const [text, setText] = useState(fmt(value));
  const parse = (t: string): number | null | undefined => {
    if (!t.trim()) return allowEmpty ? null : undefined;
    const n = Number(t.trim().replace(",", "."));
    return Number.isFinite(n) ? n : undefined;
  };
  // Keep the typed text while it already means `value`: rewriting "2." to
  // "2" would turn the next digit into "27" instead of "2.7".
  useEffect(() => {
    setText((t) => (parse(t) === value ? t : fmt(value)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);
  const tidy = (n: number) => Math.round(Math.min(range.max, Math.max(range.min, n)) / step) * step;
  return (
    <TextField fullWidth size="small" label={label} value={text} placeholder={allowEmpty ? "Auto" : undefined}
      helperText={`${range.min}–${range.max}${unit ? ` ${unit}` : ""}${allowEmpty ? ", empty = auto" : ""}`}
      sx={fieldSx}
      onChange={(e) => {
        setText(e.target.value);
        const n = parse(e.target.value);
        if (n === null || (n !== undefined && n >= range.min && n <= range.max)) onCommit(n);
      }}
      onBlur={() => {
        const n = parse(text);
        if (n === undefined) { setText(fmt(value)); return; }
        const v = n === null ? null : Number(tidy(n).toFixed(2));
        setText(fmt(v));
        onCommit(v);
      }}
      slotProps={{ inputLabel: { shrink: true }, htmlInput: { inputMode: "decimal" } }} />
  );
}

/** Which values a graph shows and in what order: a checkbox per value plus
 *  up / down buttons. Values stored in the options but absent from the
 *  data are kept and tagged. */
function NameList<T extends string>({ title, available, order, exclude, labelOf, onChange }: {
  title: string; available: readonly T[]; order: readonly T[]; exclude: readonly T[];
  labelOf?: (v: T) => string; onChange: (next: { order: T[]; exclude: T[] }) => void;
}) {
  const items = resolveOrder<T>([...new Set([...available, ...order, ...exclude])], order);
  const present = new Set(available);
  const off = new Set(exclude);
  const move = (i: number, d: -1 | 1) => {
    const next = [...items];
    [next[i], next[i + d]] = [next[i + d], next[i]];
    onChange({ order: next, exclude: [...exclude] });
  };
  const toggle = (v: T, on: boolean) =>
    onChange({ order: [...order], exclude: on ? exclude.filter((x) => x !== v) : [...exclude, v] });
  return (
    <Box sx={{ minWidth: 0 }}>
      <Typography variant="caption" sx={{ display: "block", fontSize: "0.7rem", color: "text.secondary", mb: 0.25 }}>
        {title}
      </Typography>
      {items.length === 0 && (
        <Typography variant="caption" sx={{ display: "block", color: "text.disabled", fontStyle: "italic", fontSize: "0.7rem" }}>
          None listed yet
        </Typography>
      )}
      {items.map((v, i) => {
        const name = labelOf ? labelOf(v) : v;
        return (
          <Box key={v} sx={{ display: "flex", alignItems: "center", gap: 0.25, minWidth: 0 }}>
            <Checkbox size="small" checked={!off.has(v)} onChange={(e) => toggle(v, e.target.checked)}
              sx={{ p: 0.4 }} slotProps={{ input: { "aria-label": `Include ${name}` } }} />
            <Typography variant="body2" title={name}
              sx={{ fontSize: "0.78rem", flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    color: off.has(v) ? "text.disabled" : "text.primary" }}>
              {name}
              {!present.has(v) && (
                <Typography component="span" variant="caption" sx={{ ml: 0.75, color: "text.disabled", fontSize: "0.66rem" }}>
                  {NOT_IN_DATA}
                </Typography>
              )}
            </Typography>
            <IconButton size="small" disabled={i === 0} onClick={() => move(i, -1)} aria-label={`Move ${name} up`} sx={{ p: 0.25 }}>
              <ArrowUpwardIcon sx={{ fontSize: 15 }} />
            </IconButton>
            <IconButton size="small" disabled={i === items.length - 1} onClick={() => move(i, 1)} aria-label={`Move ${name} down`} sx={{ p: 0.25 }}>
              <ArrowDownwardIcon sx={{ fontSize: 15 }} />
            </IconButton>
          </Box>
        );
      })}
    </Box>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Box component="section" aria-label={title} sx={{ border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1.1 }}>
      <Typography variant="caption" component="h3" sx={{ display: "block", fontWeight: 700, mb: 0.9, fontSize: "0.74rem" }}>
        {title}
      </Typography>
      <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 1.25, alignItems: "start" }}>
        {children}
      </Box>
    </Box>
  );
}

// ── Dialog ──────────────────────────────────────────────────

export default function IntensityPlotOptionsDialog(props: IntensityPlotOptionsDialogProps) {
  const theme = useTheme();
  const fullScreen = useMediaQuery(theme.breakpoints.down("sm"));
  // A fresh body (and so a fresh draft from the node's stored options) for
  // every opening, even one that starts while the previous close animates.
  const [session, setSession] = useState(0);
  const [wasOpen, setWasOpen] = useState(props.open);
  if (props.open !== wasOpen) {
    setWasOpen(props.open);
    if (props.open) setSession((s) => s + 1);
  }
  // Only Cancel and Escape close it: the node's Plot options icon invites a
  // double-click, whose second click would otherwise land on the backdrop
  // and close the dialog it just opened (and a stray click would lose edits).
  const onClose = (_: unknown, reason?: "backdropClick" | "escapeKeyDown") => {
    if (reason !== "backdropClick") props.onClose();
  };
  return (
    <Dialog open={props.open} onClose={onClose} maxWidth="md" fullWidth fullScreen={fullScreen}
      aria-labelledby="intensity-plot-options-title"
      slotProps={{ paper: { sx: { height: { sm: "min(820px, calc(100% - 64px))" } } } }}>
      <OptionsBody key={session} {...props} />
    </Dialog>
  );
}

function OptionsBody({ initial, context, codeState, nodeLabel, onClose, onSave }: IntensityPlotOptionsDialogProps) {
  const [draft, setDraft] = useState<IntensityPlotConfig>(
    () => normalizeIntensityPlotConfig(initial) ?? defaultIntensityPlotConfig());
  const unavailable = useMemo(
    () => Object.fromEntries(INTENSITY_GRAPH_IDS.map((id) => [id, intensityGraphUnavailableReason(id, context)])) as
      Record<IntensityGraphId, string | null>,
    [context]);
  const [selected, setSelected] = useState<IntensityGraphId>(() =>
    INTENSITY_GRAPH_IDS.find((id) => !unavailable[id] && resolveGraph(draft, id).enabled) ?? INTENSITY_GRAPH_IDS[0]);
  const [notice, setNotice] = useState<string | null>(null);
  // Bumped by the reset / copy buttons so the text fields re-read the draft.
  const [rev, setRev] = useState(0);

  const info = INTENSITY_GRAPHS.find((g) => g.id === selected) ?? INTENSITY_GRAPHS[0];
  const o = resolveGraph(draft, selected);
  const fc = isFoldChangeGraph(selected);
  const has = (k: IntensitySettingKey) => graphHasSetting(selected, k);
  const why = (k: IntensitySettingKey) => settingInactiveReason(o, k);
  const set = (patch: IntensityGraphOverride) => {
    setDraft((c) => updateIntensityGraph(c, selected, patch));
    setNotice(null);
  };
  const canApply = INTENSITY_GRAPH_IDS.some((id) => !unavailable[id] && resolveGraph(draft, id).enabled);
  const channelsAvailable = selected === "control_intensity" ? context.controlChannels : context.channels;
  const groupChoices = resolveOrder([...new Set([...context.groups, ...o.order.groups])], o.order.groups, o.exclude.groups);
  if (o.refGroup && !groupChoices.includes(o.refGroup)) groupChoices.push(o.refGroup);
  const refReason = !fc && (o.test === "none" || o.compare === "all")
    ? "Used when comparing with a reference group" : null;
  const dataNote = context.source === "unknown"
    ? "Run the Intensity picker once to list groups and channels."
    : context.source === "picker"
      ? "Listed from the Intensity picker's settings; they apply once it has run."
      : null;

  const resetOne = () => { setDraft((c) => resetIntensityGraph(c, selected)); setRev((r) => r + 1); setNotice(null); };
  const resetAll = () => {
    setDraft((c) => INTENSITY_GRAPH_IDS.reduce(resetIntensityGraph, c));
    setRev((r) => r + 1);
    setNotice("Every graph is back to the default settings.");
  };
  const copyToAll = () => {
    setDraft((c) => applyIntensityGraphToAll(c, selected));
    setRev((r) => r + 1);
    setNotice(selected === "control_intensity"
      ? "Copied to the other graphs (only settings each one offers; titles, y labels and their channel lists kept)."
      : "Copied to the other graphs (only settings each one offers; titles and y labels kept, and the control graph keeps its channel list).");
  };

  const sections: IntensityGraphInfo["section"][] = ["descriptive", "perImage", "foldChange", "control"];

  return (
    <>
      <DialogTitle id="intensity-plot-options-title" sx={{ fontSize: "1rem", py: 1.25 }}>
        📊 Plot options{nodeLabel ? ` — ${nodeLabel}` : ""}
        <Typography component="span" variant="caption" sx={{ ml: 1.5, color: "text.secondary" }}>
          Choose the graphs and how each one looks · used on the next run
        </Typography>
      </DialogTitle>
      <DialogContent dividers
        sx={{ p: 0, display: "flex", flexDirection: "column", overflow: { xs: "auto", sm: "hidden" } }}>
        {codeState === "edited" && (
          <Alert severity="warning" sx={{ m: 1.5, mb: 0, py: 0.25, fontSize: "0.75rem" }}>
            This node&apos;s R code was edited by hand. Applying replaces it with generated code; the current code
            is kept and can be restored from the node panel.
          </Alert>
        )}
        <Box sx={{ display: "flex", flexDirection: { xs: "column", sm: "row" }, flex: 1, minHeight: 0 }}>
          {/* ── Left: graphs ── */}
          <Box component="nav" aria-label="Graphs"
            sx={{ width: { sm: 280 }, flexShrink: 0, overflowY: { sm: "auto" }, p: 1.25,
                  borderRight: { sm: "1px solid" }, borderBottom: { xs: "1px solid", sm: "none" }, borderColor: "divider" }}>
            {sections.map((sec) => (
              <Box key={sec} sx={{ mb: 1 }}>
                <Typography variant="caption" component="h3"
                  sx={{ display: "block", letterSpacing: 1, fontSize: "0.6rem", textTransform: "uppercase", color: "text.secondary", fontWeight: 700, mb: 0.25 }}>
                  {SECTION_TITLE[sec]}
                </Typography>
                {INTENSITY_GRAPHS.filter((g) => g.section === sec).map((g) => {
                  const reason = unavailable[g.id];
                  const on = resolveGraph(draft, g.id).enabled;
                  const isSel = g.id === selected;
                  return (
                    <Box key={g.id} sx={{
                      display: "flex", alignItems: "flex-start", borderRadius: 1,
                      bgcolor: isSel ? "action.selected" : "transparent",
                      "&:hover": { bgcolor: isSel ? "action.selected" : "action.hover" },
                    }}>
                      <Checkbox size="small" checked={on} disabled={!!reason} sx={{ p: 0.5, mt: 0.1 }}
                        onChange={(e) => setDraft((c) => updateIntensityGraph(c, g.id, { enabled: e.target.checked }))}
                        slotProps={{ input: { "aria-label": `Draw ${g.label}` } }} />
                      <ButtonBase onClick={() => setSelected(g.id)} aria-current={isSel ? "true" : undefined}
                        sx={{ flex: 1, minWidth: 0, display: "block", textAlign: "left", py: 0.5, pr: 0.75, borderRadius: 1 }}>
                        <Typography variant="body2" component="span"
                          sx={{ display: "block", fontSize: "0.8rem", fontWeight: isSel ? 700 : 500, color: reason ? "text.disabled" : "text.primary" }}>
                          {g.label}
                        </Typography>
                        <Typography variant="caption" component="span"
                          sx={{ display: "block", color: "text.secondary", fontSize: "0.68rem", lineHeight: 1.3 }}>
                          {g.description}
                        </Typography>
                        {reason && (
                          <Typography variant="caption" component="span"
                            sx={{ display: "block", color: warnColor, fontSize: "0.66rem", lineHeight: 1.3 }}>
                            {reason}
                          </Typography>
                        )}
                      </ButtonBase>
                    </Box>
                  );
                })}
              </Box>
            ))}
          </Box>

          {/* ── Right: the selected graph's settings ── */}
          <Box component="section" aria-label={`${info.label} settings`}
            sx={{ flex: 1, minWidth: 0, overflowY: { sm: "auto" }, p: 1.5, display: "flex", flexDirection: "column", gap: 1.25 }}>
            <Box sx={{ display: "flex", alignItems: "flex-start", gap: 1, flexWrap: "wrap" }}>
              <Box sx={{ flex: 1, minWidth: 180 }}>
                <Typography variant="subtitle2" component="h2" sx={{ fontWeight: 700, lineHeight: 1.3 }}>{info.label}</Typography>
                <Typography variant="caption" sx={{ display: "block", color: "text.secondary" }}>{info.description}</Typography>
                {(unavailable[selected] || !o.enabled) && (
                  <Typography variant="caption" sx={{ display: "block", color: warnColor }}>
                    {unavailable[selected]
                      ? `Not drawn for the current data: ${unavailable[selected]}`
                      : "Not drawn: tick it on the left to include it"}
                  </Typography>
                )}
                {/* Always present, so screen readers announce what the buttons did. */}
                <Typography variant="caption" role="status" sx={{ display: "block", color: okColor }}>{notice}</Typography>
              </Box>
              <Box sx={{ display: "flex", gap: 0.75, flexWrap: "wrap" }}>
                <Tooltip describeChild title="Copy these settings to every other graph, where they apply. Titles, y labels and which graphs are drawn are not copied.">
                  <Button size="small" variant="outlined" onClick={copyToAll}>Apply these settings to all graphs</Button>
                </Tooltip>
                <Button size="small" onClick={resetOne}>Reset this graph</Button>
              </Box>
            </Box>

            {has("chart") && (
              <Section title="Chart">
                <ToggleSetting label="Chart type" value={o.chart}
                  options={CHOICES.chart.map((v) => ({ value: v, label: CHART_LABEL[v] }))}
                  onChange={(chart) => set({ chart })} />
                <SelectSetting label="Error bars" value={o.error} reason={why("error")}
                  options={CHOICES.error.map((v) => ({ value: v, label: ERROR_LABEL[v] }))}
                  onChange={(error) => set({ error })} />
                <CheckSetting label="Show each value" checked={o.points} onChange={(points) => set({ points })} />
              </Section>
            )}

            {has("refGroup") && (
              <Section title={fc ? "Baseline" : "Statistics"}>
                {has("test") && (
                  <SelectSetting label="Test" value={o.test}
                    help={o.test === "tukey" && o.compare === "ref"
                      ? "Tukey p-values come from all pairs (conservative); only the reference pairs are shown."
                      : undefined}
                    options={CHOICES.test.map((v) => ({ value: v, label: TEST_LABEL[v] }))}
                    onChange={(test) => set({ test })} />
                )}
                {has("compare") && (
                  <ToggleSetting label="Compare" value={o.compare} reason={why("compare")}
                    options={CHOICES.compare.map((v) => ({ value: v, label: COMPARE_LABEL[v] }))}
                    onChange={(compare) => set({ compare })} />
                )}
                <SelectSetting label={fc ? "Baseline group" : "Reference group"} value={o.refGroup ?? AUTO_REF}
                  reason={refReason}
                  help={fc ? "Fold change is log2(group mean / baseline mean)" : undefined}
                  options={[
                    { value: AUTO_REF, label: "Automatic (first group shown)" },
                    ...groupChoices.map((g) => ({
                      value: g,
                      label: context.groups.includes(g) ? g : `${g} (${NOT_IN_DATA})`,
                    })),
                  ]}
                  onChange={(v) => set({ refGroup: v === AUTO_REF ? null : v })} />
                {has("sig") && (
                  <ToggleSetting label="Significance" value={o.sig} reason={why("sig")}
                    options={CHOICES.sig.map((v) => ({ value: v, label: SIG_LABEL[v] }))}
                    onChange={(sig) => set({ sig })} />
                )}
                {has("hideNs") && (
                  <CheckSetting label="Hide non-significant brackets" checked={o.hideNs} reason={why("hideNs")}
                    onChange={(hideNs) => set({ hideNs })} />
                )}
              </Section>
            )}

            <Section title="Data">
              {dataNote && (
                <Typography variant="caption" sx={{ gridColumn: "1 / -1", color: "text.secondary" }}>{dataNote}</Typography>
              )}
              <NameList title="Groups" available={context.groups} order={o.order.groups} exclude={o.exclude.groups}
                onChange={(n) => set({ order: { ...o.order, groups: n.order }, exclude: { ...o.exclude, groups: n.exclude } })} />
              <NameList title={selected === "control_intensity" ? "Control channels" : "Channels"}
                available={channelsAvailable} order={o.order.channels} exclude={o.exclude.channels}
                onChange={(n) => set({ order: { ...o.order, channels: n.order }, exclude: { ...o.exclude, channels: n.exclude } })} />
              <NameList<CompId> title="Compartments" available={context.compartments}
                order={o.order.compartments} exclude={o.exclude.compartments}
                labelOf={(c) => INTENSITY_COMPARTMENT_LABELS[c]}
                onChange={(n) => set({
                  order: { ...o.order, compartments: n.order }, exclude: { ...o.exclude, compartments: n.exclude },
                })} />
            </Section>

            <Section title="Layout & axes">
              <ToggleSetting label="Compartment layout" value={o.layout} help={LAYOUT_HELP[o.layout]}
                options={CHOICES.layout.map((v) => ({ value: v, label: LAYOUT_LABEL[v] }))}
                onChange={(layout) => set({ layout })} />
              <SelectSetting label="Y axis range" value={o.yScale}
                options={CHOICES.yScale.map((v) => ({ value: v, label: YSCALE_LABEL[v] }))}
                onChange={(yScale) => set({ yScale })} />
              {has("yZero") && (
                <CheckSetting label="Start the y axis at 0" checked={yAxisStartsAtZero(o)} reason={why("yZero")}
                  onChange={(yZero) => set({ yZero })} />
              )}
              {has("yLog") && (
                <CheckSetting label="Log scale y axis" checked={o.yLog} reason={why("yLog")}
                  onChange={(yLog) => set({ yLog })} />
              )}
              <ToggleSetting label="X label angle" value={o.xAngle}
                options={INTENSITY_X_ANGLES.map((v) => ({ value: v, label: `${v}°` }))}
                onChange={(xAngle) => set({ xAngle })} />
            </Section>

            <Section title="Text & style">
              <TextSetting key={`title-${selected}-${rev}`} label="Title" value={o.title}
                onCommit={(title) => set({ title })} />
              <TextSetting key={`ylab-${selected}-${rev}`} label="Y axis label" value={o.yLabel}
                onCommit={(yLabel) => set({ yLabel })} />
              <SelectSetting label="Colours" value={o.palette}
                options={CHOICES.palette.map((v) => ({
                  value: v,
                  label: (
                    <Box component="span" sx={{ display: "inline-flex", alignItems: "center", gap: 0.75 }}>
                      <Box component="span" aria-hidden sx={{ display: "inline-flex" }}>
                        {PALETTE_SWATCH[v].map((c) => (
                          <Box key={c} component="span" sx={{ width: 10, height: 10, bgcolor: c, border: "1px solid", borderColor: "divider" }} />
                        ))}
                      </Box>
                      {PALETTE_LABEL[v]}
                    </Box>
                  ),
                }))}
                onChange={(palette) => set({ palette })} />
              <ToggleSetting label="Legend" value={o.legend}
                options={CHOICES.legend.map((v) => ({ value: v, label: LEGEND_LABEL[v] }))}
                onChange={(legend) => set({ legend })} />
              <NumberSetting key={`font-${selected}-${rev}`} label="Font size" value={o.fontSize} unit="pt"
                range={INTENSITY_FONT_RANGE} step={0.5} onCommit={(v) => { if (v !== null) set({ fontSize: v }); }} />
            </Section>

            <Section title="Size">
              <NumberSetting key={`pw-${selected}-${rev}`} label="Panel width" value={o.panelW} unit="in" allowEmpty
                range={INTENSITY_PANEL_RANGE} step={0.01} onCommit={(panelW) => set({ panelW })} />
              <NumberSetting key={`ph-${selected}-${rev}`} label="Panel height" value={o.panelH} unit="in" allowEmpty
                range={INTENSITY_PANEL_RANGE} step={0.01} onCommit={(panelH) => set({ panelH })} />
            </Section>
          </Box>
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: 2, py: 1.25, gap: 1, flexWrap: "wrap" }}>
        <Tooltip describeChild title="Every graph's settings back to the defaults (which graphs are drawn is kept)">
          <Button size="small" onClick={resetAll}>Reset all</Button>
        </Tooltip>
        <Box sx={{ flex: 1 }} />
        {!canApply && (
          <Typography variant="caption" sx={{ color: warnColor }}>
            Tick at least one graph this data can draw.
          </Typography>
        )}
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" disabled={!canApply} onClick={() => onSave(draft)}>Apply</Button>
      </DialogActions>
    </>
  );
}
