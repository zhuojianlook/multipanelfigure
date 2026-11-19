/* ──────────────────────────────────────────────────────────
   intensityPlotRuntime — the fixed R half of the "Plot intensities"
   node. generateIntensityPlotR (intensityPlotOptions.ts) puts a `.ipo`
   options list in front of this text and one emit line per graph after
   it; nothing is interpolated in here.

   Rules this code must keep:
   - Pure ASCII. R may run under LC_CTYPE=C, so non-ASCII text is written
     as \U{...} escapes.
   - No ggplot object is created before its device is open. ggplot() and
     `+` both update last_plot(), and the harness snapshots last_plot()
     when the NEXT device opens, so an early build would file the wrong
     plot under the previous index. Hence .ipo_prepare (data only) runs
     first, and .ipo_plot builds the plot after the device is open.
   - Never name a group, channel or compartment in a string key: split on
     integer codes, which also keeps Tukey's row names unambiguous.
   - The helper names used by the harness must not appear here followed
     by a quoted file name: the node's output chips are parsed from the
     code by regex.
   - Group, channel and compartment names are compared as the table spells
     them (the dialog lists them that way), so the table is read again as
     text from the prelude's .mpfig_csv_text (see rPrelude.ts).
   ────────────────────────────────────────────────────────── */

export const INTENSITY_PLOT_RUNTIME_R = String.raw`suppressWarnings(suppressMessages({ library(ggplot2); library(ggprism) }))

.ipo_COMP <- c("whole_image", "whole_cell", "nucleus", "cytoplasm")
.ipo_COMP_LABEL <- c("Whole image", "Whole cell", "Nuclear", "Cytoplasm")
.ipo_UNASSIGNED <- "(unassigned)"
.ipo_DASH <- " \U{2014} "
.ipo_SEED <- 1L
.ipo_DODGE <- 0.8
.ipo_PAL <- list(
  set2  = c("#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"),
  okabe = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"),
  dark  = c("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666"),
  grey  = c("#CCCCCC", "#969696", "#636363", "#252525"))
.ipo_env <- new.env(parent = emptyenv())
.ipo_env$n <- 0L
.ipo_env$short <- NULL

# -- Input table ------------------------------------------------------
.ipo_data <- local({
  raw <- if (exists("inputs") && length(inputs) > 0) inputs[[1]] else data.frame()
  # inputs[[1]] went through read.csv's type conversion, which rewrites names
  # that look like numbers or logicals ("0.10" -> 0.1, "01" -> 1, "T" -> TRUE)
  # and would then match no stored option. Read the same text again with
  # every column as text; the numeric columns are converted below.
  txt <- if (exists(".mpfig_csv_text") && length(.mpfig_csv_text) > 0L) .mpfig_csv_text[[1]] else NULL
  if (is.character(txt) && length(txt) == 1L) {
    as_text <- tryCatch(utils::read.csv(text = txt, colClasses = "character", stringsAsFactors = FALSE),
                        error = function(e) NULL)
    if (is.data.frame(as_text) && nrow(as_text) == nrow(raw)) raw <- as_text
  }
  n <- nrow(raw)
  # read.csv turns the text "NA" into a missing value. The picker always
  # writes a name, so a missing group or channel really is called "NA".
  text_col <- function(name, absent, blank = absent, missing = blank) {
    v <- if (name %in% names(raw)) as.character(raw[[name]]) else rep(absent, n)
    v[is.na(v)] <- missing
    v[!nzchar(v)] <- blank
    v
  }
  num_col <- function(name, absent) {
    if (name %in% names(raw)) suppressWarnings(as.numeric(raw[[name]])) else rep(absent, n)
  }
  ctl <- if ("is_control" %in% names(raw)) as.logical(raw[["is_control"]]) else rep(FALSE, n)
  ctl[is.na(ctl)] <- FALSE
  src <- text_col("source", NA_character_, NA_character_, NA_character_)
  d <- data.frame(
    source = src,
    group = text_col("group", "all", .ipo_UNASSIGNED, "NA"),
    channel = text_col("channel", "Channel", "Channel", "NA"),
    compartment = text_col("compartment", "whole_image"),
    is_control = ctl,
    mean_intensity = num_col("mean_intensity", 0),
    mean_intensity_norm = num_col("mean_intensity_norm", NA_real_),
    stringsAsFactors = FALSE)
  unknown <- !(d$compartment %in% .ipo_COMP)
  if (any(unknown)) {
    message("Plot options: ignored ", sum(unknown), " row(s) with an unknown compartment")
    d <- d[!unknown, , drop = FALSE]
  }
  d[is.finite(d$mean_intensity), , drop = FALSE]
})
.ipo_groups <- unique(.ipo_data$group)
.ipo_cellpose <- any(.ipo_data$compartment %in% c("whole_cell", "nucleus", "cytoplasm"))
.ipo_has_norm <- any(is.finite(.ipo_data$mean_intensity_norm))
.ipo_has_source <- any(!is.na(.ipo_data$source))

.ipo_SPEC <- list(
  channel_intensity = list(kind = "plain", col = "mean_intensity", control = FALSE, per_image = FALSE,
                           ylab = "Mean intensity (BG-corrected)"),
  channel_intensity_norm = list(kind = "plain", col = "mean_intensity_norm", control = FALSE, per_image = FALSE,
                                ylab = "Mean intensity / per-image control"),
  channel_intensity_per_image = list(kind = "plain", col = "mean_intensity", control = FALSE, per_image = TRUE,
                                     ylab = "Per-image mean intensity"),
  channel_intensity_norm_per_image = list(kind = "plain", col = "mean_intensity_norm", control = FALSE, per_image = TRUE,
                                          ylab = "Per-image normalised intensity"),
  channel_intensity_fc = list(kind = "fc", col = "mean_intensity", control = FALSE, per_image = FALSE),
  channel_intensity_norm_fc = list(kind = "fc", col = "mean_intensity_norm", control = FALSE, per_image = FALSE),
  control_intensity = list(kind = "plain", col = "mean_intensity", control = TRUE, per_image = FALSE,
                           ylab = "Mean intensity (BG-corrected)"))

.ipo_skip <- function(id, why) {
  message("Plot options: skipped ", id, " (", why, ")")
  NULL
}

# Same rule as resolveOrder in intensityPlotOptions.ts.
.ipo_pick <- function(present, order, exclude) {
  setdiff(c(intersect(order, present), setdiff(present, order)), exclude)
}

.ipo_per_image <- function(d) {
  d <- d[!is.na(d$source), , drop = FALSE]
  if (!nrow(d)) return(d)
  key <- paste(match(d$source, unique(d$source)), match(d$group, unique(d$group)),
               match(d$channel, unique(d$channel)), match(d$compartment, unique(d$compartment)))
  parts <- split(seq_len(nrow(d)), key)
  first <- vapply(parts, function(ii) ii[1], 1L, USE.NAMES = FALSE)
  y <- vapply(parts, function(ii) mean(d$y[ii]), 0, USE.NAMES = FALSE)
  out <- d[first, , drop = FALSE]
  out$y <- y
  rownames(out) <- NULL
  out
}

.ipo_quote <- function(x) paste0("\"", x, "\"", collapse = ", ")

# Names in a graph's options that the table does not have do nothing; say so,
# so a misspelt or since-renamed group is not mistaken for an applied option.
.ipo_note_unmatched <- function(id, o, sp) {
  rows <- .ipo_data$is_control == sp$control
  check <- function(what, wanted, present) {
    miss <- setdiff(unique(wanted), present)
    if (length(miss)) message("Plot options: ", id, ": ", what, " not in the table, ignored: ", .ipo_quote(miss))
  }
  check("group(s)", c(o$exclude_groups, o$order_groups), .ipo_groups)
  check("channel(s)", c(o$exclude_channels, o$order_channels), unique(.ipo_data$channel[rows]))
  check("compartment(s)", c(o$exclude_compartments, o$order_compartments), unique(.ipo_data$compartment))
}

# Rows of one graph, filtered and ordered, with group / channel /
# compartment as factors. Returns a character reason when nothing is left.
.ipo_frame <- function(id, o, sp, log_y) {
  .ipo_note_unmatched(id, o, sp)
  d <- .ipo_data[.ipo_data$is_control == sp$control, , drop = FALSE]
  d$y <- d[[sp$col]]
  d <- d[is.finite(d$y), , drop = FALSE]
  if (sp$per_image) d <- .ipo_per_image(d)
  if (!nrow(d)) return("no rows with a value for this graph")
  groups <- .ipo_pick(.ipo_groups, o$order_groups, o$exclude_groups)
  channels <- .ipo_pick(unique(d$channel), o$order_channels, o$exclude_channels)
  comps <- .ipo_pick(.ipo_COMP[.ipo_COMP %in% d$compartment], o$order_compartments, o$exclude_compartments)
  d <- d[d$group %in% groups & d$channel %in% channels & d$compartment %in% comps, , drop = FALSE]
  if (log_y) {
    bad <- d$y <= 0
    if (any(bad)) {
      message("Plot options: ", id, ": log scale dropped ", sum(bad), " value(s) at or below zero")
      d <- d[!bad, , drop = FALSE]
    }
  }
  if (!nrow(d)) return("every row is excluded by the graph's options")
  channels <- channels[channels %in% d$channel]
  comps <- comps[comps %in% d$compartment]
  d$group <- factor(d$group, levels = groups)
  d$channel <- factor(d$channel, levels = channels)
  d$compartment <- factor(d$compartment, levels = comps, labels = .ipo_COMP_LABEL[match(comps, .ipo_COMP)])
  d
}

.ipo_summary <- function(d, err) {
  ng <- nlevels(d$group)
  nc <- nlevels(d$channel)
  key <- as.integer(d$group) + ng * (as.integer(d$channel) - 1L) + ng * nc * (as.integer(d$compartment) - 1L)
  one <- function(ii) {
    y <- d$y[ii]
    n <- length(y)
    m <- mean(y)
    s <- if (n > 1) stats::sd(y) else 0
    e <- switch(err,
                sd = s,
                sem = s / sqrt(n),
                ci = if (n > 1) stats::qt(0.975, n - 1) * s / sqrt(n) else 0,
                0)
    c(first = ii[1], mean = m, n = n, e = e)
  }
  m <- t(vapply(split(seq_len(nrow(d)), key), one, numeric(4)))
  i <- as.integer(m[, "first"])
  data.frame(group = d$group[i], channel = d$channel[i], compartment = d$compartment[i],
             mean = m[, "mean"], n = m[, "n"], lo = m[, "mean"] - m[, "e"], hi = m[, "mean"] + m[, "e"],
             row.names = NULL)
}

# -- Statistics -------------------------------------------------------
# gi: integer group codes. Groups are recoded g1..gN before the model, so
# group names can contain "-" and TukeyHSD's row names are never split.
.ipo_pairs <- function(y, gi, ng, test, ref_i) {
  ok <- which(tabulate(gi, ng) >= 2L)
  if (length(ok) < 2L) return(NULL)
  cb <- utils::combn(length(ok), 2L)
  if (!is.null(ref_i)) {
    cb <- cb[, ok[cb[1, ]] == ref_i | ok[cb[2, ]] == ref_i, drop = FALSE]
    if (!ncol(cb)) return(NULL)
  }
  keep <- gi %in% ok
  y <- y[keep]
  code <- match(gi[keep], ok)
  if (test == "tukey") {
    df <- data.frame(y = y, g = factor(paste0("g", code), levels = paste0("g", seq_along(ok))))
    tk <- tryCatch(stats::TukeyHSD(stats::aov(y ~ g, data = df))[["g"]], error = function(e) NULL)
    if (is.null(tk)) return(NULL)
    p <- as.numeric(tk[paste0("g", cb[2, ], "-g", cb[1, ]), "p adj"])
  } else {
    p <- vapply(seq_len(ncol(cb)), function(k) {
      a <- y[code == cb[1, k]]
      b <- y[code == cb[2, k]]
      tryCatch(
        if (test == "welch") stats::t.test(a, b)$p.value
        else suppressWarnings(stats::wilcox.test(a, b, exact = FALSE)$p.value),
        error = function(e) NA_real_)
    }, 0)
    p <- stats::p.adjust(p, "holm")
  }
  data.frame(a = ok[cb[1, ]], b = ok[cb[2, ]], p = p)
}

.ipo_label_p <- function(p, mode) {
  if (mode == "p") return(ifelse(p < 0.001, "p<0.001", paste0("p=", formatC(p, format = "f", digits = 3))))
  ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "ns")))
}

.ipo_ref_index <- function(id, o, d) {
  if (o$compare != "ref") return(NULL)
  lv <- levels(d$group)
  have <- sort(unique(as.integer(d$group)))
  wanted <- if (is.null(o$ref_group)) NA_integer_ else match(o$ref_group, lv)
  if (!is.na(wanted) && wanted %in% have) return(wanted)
  if (!is.null(o$ref_group))
    message("Plot options: ", id, ": reference group ", .ipo_quote(o$ref_group),
            " has no data in this graph; compared with ", .ipo_quote(lv[have[1]]), " instead")
  have[1]
}

.ipo_brackets <- function(g) {
  o <- g$o; d <- g$d; s <- g$s
  empty <- list(df = NULL, any_sig = FALSE, tiers = 0L)
  if (o$test == "none") return(empty)
  ch_i <- as.integer(d$channel); co_i <- as.integer(d$compartment); g_i <- as.integer(d$group)
  use_raw <- o$points || g$chart %in% c("box", "violin")
  use_sum <- g$chart %in% c("bar", "mean")
  sum_y <- if (o$error != "none") s$hi else s$mean
  vy <- c(if (use_raw) d$y, if (use_sum) sum_y)
  vc <- c(if (use_raw) ch_i, if (use_sum) as.integer(s$channel))
  vo <- c(if (use_raw) co_i, if (use_sum) as.integer(s$compartment))
  ng <- nlevels(d$group)
  ref_i <- .ipo_ref_index(g$id, o, d)
  step_scale <- max(1, o$font_size / 11)
  out <- list(); any_sig <- FALSE; tiers <- 0L
  for (ci in seq_len(nlevels(d$channel))) {
    in_ch <- vc == ci
    for (oi in seq_len(nlevels(d$compartment))) {
      rows <- which(ch_i == ci & co_i == oi)
      if (!length(rows)) next
      t <- .ipo_pairs(d$y[rows], g_i[rows], ng, o$test, ref_i)
      if (is.null(t)) next
      t <- t[is.finite(t$p), , drop = FALSE]
      if (!nrow(t)) next
      if (any(t$p < 0.05)) any_sig <- TRUE
      if (o$hide_ns) t <- t[t$p < 0.05, , drop = FALSE]
      if (!nrow(t)) next
      t <- t[order(t$b - t$a, t$a), , drop = FALSE]
      scope <- if (o$y_scale == "independent" && !g$dodged) in_ch & vo == oi else in_ch
      if (!any(scope)) next
      top <- max(vy[scope]); low <- min(vy[scope])
      k <- seq_len(nrow(t))
      # Each tier is the same share of the axis on both scales: 11% of the
      # data's span, in decades on a log axis.
      if (g$log_y) {
        span <- log10(top) - log10(low)
        span <- if (!is.finite(span) || span <= 0) 0.3 else max(span, 0.1)
        yp <- 10 ^ (log10(top) + 0.11 * step_scale * span * k)
      } else {
        base <- if (o$y_zero || g$chart == "bar") min(0, low) else low
        span <- top - base
        if (!is.finite(span) || span <= 0) span <- max(abs(top), 1)
        yp <- top + 0.11 * step_scale * span * k
      }
      if (g$dodged) {
        # Dodging packs the groups a cell has, so the slots depend on which
        # groups are present in this channel x compartment.
        present <- sort(unique(g_i[rows]))
        m <- length(present)
        off <- function(i) oi + (match(i, present) - (m + 1) / 2) * .ipo_DODGE / m
        xmin <- off(t$a); xmax <- off(t$b)
      } else {
        xmin <- levels(d$group)[t$a]; xmax <- levels(d$group)[t$b]
      }
      tiers <- max(tiers, nrow(t))
      out[[length(out) + 1L]] <- data.frame(
        channel = factor(levels(d$channel)[ci], levels = levels(d$channel)),
        compartment = factor(levels(d$compartment)[oi], levels = levels(d$compartment)),
        xmin = xmin, xmax = xmax, label = .ipo_label_p(t$p, o$sig), y.position = yp,
        stringsAsFactors = FALSE)
    }
  }
  list(df = if (length(out)) do.call(rbind, out) else NULL, any_sig = any_sig, tiers = tiers)
}

# -- Layout -----------------------------------------------------------
.ipo_shade <- function(cols, to, amount) {
  m <- grDevices::col2rgb(cols) / 255
  m <- m * (1 - amount) + amount * to
  grDevices::rgb(m[1, ], m[2, ], m[3, ])
}

# Greys are a ramp. The other palettes keep their own colours; with more
# groups than colours they repeat darker, then lighter (blending neighbours
# would give look-alike shades and lose Okabe-Ito's colour-blind safety).
.ipo_palette <- function(name, n) {
  n <- max(n, 1L)
  if (identical(name, "grey")) return(grDevices::colorRampPalette(.ipo_PAL$grey)(n))
  base <- .ipo_PAL[[name]]
  if (is.null(base)) base <- .ipo_PAL$set2
  if (n <= length(base)) return(base[seq_len(n)])
  rep_len(c(base, .ipo_shade(base, 0, 0.45), .ipo_shade(base, 1, 0.55)), n)
}

# A group keeps its colour across the node's graphs whichever groups a graph
# leaves out: colours follow the graph's order of ALL groups.
.ipo_colours <- function(o, shown) {
  full <- .ipo_pick(.ipo_groups, o$order_groups, character(0))
  out <- .ipo_palette(o$palette, length(full))[match(shown, full)]
  out[is.na(out)] <- "#999999"
  out
}

# A pixel size the PNG device keeps. It works from inches (px / res) and
# truncates on the way back, so about 1 width in 20 comes out one pixel
# smaller. A re-render asks for the size of the saved file, and would then
# lay the plot out on a slightly different canvas.
.ipo_px <- function(x) {
  x <- as.integer(round(min(6000, max(900, x))))
  while (trunc(x / 300 * 300) != x) x <- x + 1L
  x
}

# Text width in em, from Helvetica's character widths (the PNG device's sans
# font is metrically close). Measuring with grid would need an open device,
# and none may be opened before the graph's own.
.ipo_EM <- stats::setNames(c(
  667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611,
  556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500,
  rep(556, 10), 278, 333, 584, 278, 278, 333, 333, 556, 278, 278, 889, 667) / 1000,
  strsplit("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -+.,()_/:%&", "")[[1]])
.ipo_em <- function(x) {
  vapply(strsplit(as.character(x), ""), function(cs) {
    w <- .ipo_EM[cs]
    w[is.na(w)] <- 0.7
    sum(w)
  }, 0)
}

# The text size the plot is drawn at: a collage re-render may force a base
# font size (the harness's mpfig_base_size option) over the graph's own.
.ipo_pt <- function(o) {
  pt <- suppressWarnings(as.numeric(getOption("mpfig_base_size", o$font_size)))
  if (length(pt) != 1L || !is.finite(pt) || pt <= 0) o$font_size else pt
}

.ipo_dims <- function(o, dodged, ng, nch, nco, tiers, group_names) {
  fs <- max(1, o$font_size / 11)
  lab <- unlist(strsplit(.mpfig_wrap(group_names, 13), "\n", fixed = TRUE))
  lab_chars <- if (length(lab)) max(nchar(lab)) else 1L
  # Width one x label needs. A wrapped label slanted at angle a is
  # (lines * line height) thick, which takes thickness / sin(a) along the axis.
  x_lab <- strsplit(.mpfig_wrap(group_names, 14), "\n", fixed = TRUE)
  x_lines <- if (length(x_lab)) max(lengths(x_lab)) else 1L
  slot <- if (o$x_angle == 0) max(105, min(14L, lab_chars) * 17)
          else max(105, x_lines * 38 / sin(o$x_angle * pi / 180))
  pw <- fs * (if (dodged) 300 + 95 * ng * nco else max(360, 150 + slot * ng))
  ph <- fs * max(620, 432 * (1 + 0.11 * tiers))
  if (!is.null(o$panel_w)) pw <- o$panel_w * 300
  if (!is.null(o$panel_h)) ph <- o$panel_h * 300
  cols <- if (dodged) 1L else if (nco > 1L) nco else nch
  rows <- if (dodged || nco > 1L) nch else 1L
  w <- 260 * fs + cols * pw + (if (o$legend == "right") 420 * fs else 0)
  h <- 420 * fs + rows * ph - (if (o$legend %in% c("none", "right")) 180 * fs else 0)
  # A bottom / top legend is one row of keys after the "Group" title.
  # Widths in px at 300 dpi, measured on theme_prism's legend: a key is
  # 46 + 7.8 pt plus its label (legend text is 0.73 pt, 300 / 72 px per pt,
  # 8% spare); the title, the legend's own margins and the plot margins
  # take 108 + 15.2 pt.
  pt <- .ipo_pt(o)
  lab_em <- if (length(lab)) max(.ipo_em(lab)) else 0.6
  key_w <- 46 + 7.8 * pt + lab_em * 0.73 * pt * 300 / 72 * 1.08
  fixed_w <- 108 + 15.2 * pt
  legend_rows <- 1L
  if (o$legend %in% c("bottom", "top")) {
    w <- max(w, fixed_w + key_w)
    per_row <- max(1, floor((min(6000, max(900, w)) - fixed_w) / key_w))
    legend_rows <- as.integer(ceiling(max(ng, 1L) / per_row))
    h <- h + (legend_rows - 1L) * 90 * fs
  }
  if (o$legend == "right") h <- max(h, (200 + 95 * ng) * fs)
  list(w = .ipo_px(w), h = .ipo_px(h), legend_rows = legend_rows)
}

.ipo_facet <- function(g) {
  nco <- nlevels(g$d$compartment)
  free <- if (g$o$y_scale == "same") "fixed" else "free_y"
  if (g$dodged) return(facet_wrap(~ channel, ncol = 1, scales = free, labeller = .mpfig_labeller(24)))
  if (nco > 1L && g$o$y_scale == "independent")
    return(facet_wrap(vars(channel, compartment), ncol = nco, scales = "free_y", drop = FALSE,
                      labeller = .mpfig_labeller(14)))
  if (nco > 1L)
    return(facet_grid(channel ~ compartment, scales = free, switch = "y", labeller = .mpfig_labeller(14)))
  facet_wrap(~ channel, nrow = 1, scales = free, labeller = .mpfig_labeller(14))
}

.ipo_style <- function(p, g) {
  o <- g$o
  a <- o$x_angle
  wrap_legend <- function(z) .mpfig_wrap(z, 13)
  guide <- if (o$legend == "none") "none"
           else if (o$legend == "right") guide_legend(ncol = 1)
           else guide_legend(nrow = g$legend_rows, byrow = TRUE)
  p <- p + scale_x_discrete(limits = g$x_limits, drop = FALSE, labels = function(z) .mpfig_wrap(z, 14))
  if (identical(g$chart, "mean")) {
    p <- p +
      scale_colour_manual(values = g$colours, limits = g$legend_limits, drop = FALSE,
                          labels = wrap_legend, guide = guide) +
      scale_fill_manual(values = g$colours, limits = g$legend_limits, drop = FALSE, guide = "none")
  } else {
    p <- p + scale_fill_manual(values = g$colours, limits = g$legend_limits, drop = FALSE,
                               labels = wrap_legend, guide = guide)
  }
  p + theme_prism(base_size = o$font_size) +
    theme(legend.position = o$legend,
          legend.title = element_text(size = rel(0.82)),
          legend.text = element_text(size = rel(0.73)),
          axis.text.x = element_text(angle = a, size = rel(0.82),
                                     hjust = if (a > 0) 1 else 0.5,
                                     vjust = if (a == 90) 0.5 else 1),
          strip.placement = "outside",
          plot.margin = margin(16, 14, 8, 12),
          plot.title.position = "plot",
          plot.title = element_text(size = rel(1), face = "bold", lineheight = 1.05, margin = margin(b = 8),
                                    colour = if (isTRUE(g$flag)) "#b03030" else "grey25")) +
    .ipo_facet(g) +
    (if (identical(g$chart, "mean")) labs(colour = "Group") else labs(fill = "Group")) +
    labs(x = NULL, y = .mpfig_wrap(g$ylab, 28),
         title = .mpfig_fit(g$title, width_in = g$w / 300, cpi = 9 * 11 / max(11, .ipo_pt(o))))
}

# -- Preparing a graph (data only: see the note at the top of the file) --
# " - Nuclear" when the table has several compartments and the graph shows
# only one: no strip names it then, so the title must.
.ipo_comp_note <- function(comp) {
  if (nlevels(comp) == 1L && length(unique(.ipo_data$compartment)) > 1L) paste0(.ipo_DASH, levels(comp)) else ""
}

.ipo_default_title <- function(id, comp, flagged) {
  cell <- if (.ipo_cellpose) paste0(.ipo_DASH, "per-CELL (n = cells; descriptive)") else ""
  head <- if (nlevels(comp) > 1L) "Channel intensities by compartment" else "Channel intensities"
  note <- .ipo_comp_note(comp)
  switch(id,
    channel_intensity = paste0(head, note, cell),
    channel_intensity_norm = paste0("Channel intensities", note, .ipo_DASH, "NORMALISED per image to control channel", cell),
    channel_intensity_per_image = paste0(head, note, .ipo_DASH, "PER-IMAGE means (n = images; publication-grade)"),
    channel_intensity_norm_per_image = paste0("Channel intensities", note, .ipo_DASH,
                                              "NORMALISED, PER-IMAGE means (n = images; publication-grade)"),
    control_intensity = paste0("Control channel", note, .ipo_DASH,
                               if (flagged) "DIFFERS between groups (groups may not be comparable)"
                               else "sanity check (NS expected)"),
    id)
}

.ipo_unavailable <- function(sp) {
  rows <- .ipo_data$is_control == sp$control
  if (!any(rows)) return(if (sp$control) "no control channel rows in the table" else "no analysis channel rows in the table")
  if (sp$col == "mean_intensity_norm" && !.ipo_has_norm) return("the table has no normalised values")
  if (sp$per_image && !.ipo_cellpose) return("simple mode already measures one value per image")
  if (sp$per_image && !.ipo_has_source) return("the table has no source column")
  NULL
}

.ipo_common <- function(g) {
  d <- g$d
  dims <- .ipo_dims(g$o, g$dodged, length(g$legend_limits), nlevels(d$channel), nlevels(d$compartment),
                    g$tiers, g$legend_limits)
  g$w <- dims$w; g$h <- dims$h; g$legend_rows <- dims$legend_rows
  g
}

.ipo_prepare_plain <- function(id, o, sp) {
  log_y <- isTRUE(o$y_log) && o$chart != "bar"
  o$y_zero <- o$y_zero && !log_y
  d <- .ipo_frame(id, o, sp, log_y)
  if (is.character(d)) return(.ipo_skip(id, d))
  s <- .ipo_summary(d, if (o$chart %in% c("bar", "mean")) o$error else "none")
  if (o$chart == "bar" && all(d$y >= 0)) s$lo <- pmax(s$lo, 0)
  # A log axis cannot show mean - spread <= 0. Draw only the upper half
  # there rather than invent a lower end.
  if (log_y && any(s$lo <= 0)) {
    cut <- s$lo <= 0
    message("Plot options: ", id, ": ", sum(cut), " error bar(s) drawn upward only (mean minus ",
            switch(o$error, sd = "SD", sem = "SEM", ci = "the 95% CI", "spread"),
            " is at or below 0, which a log axis cannot show)")
    s$lo[cut] <- s$mean[cut]
  }
  chart <- o$chart
  if (chart == "violin" && min(s$n) < 3) {
    message("Plot options: ", id, " drew box plots (a violin needs at least 3 values per group)")
    chart <- "box"
  }
  if (chart %in% c("box", "violin")) o$error <- "none"
  dodged <- o$layout == "dodged" && nlevels(d$compartment) > 1L
  g <- list(kind = "plain", id = id, o = o, d = d, s = s, chart = chart, dodged = dodged, log_y = log_y)
  br <- .ipo_brackets(g)
  g$br <- br$df
  g$tiers <- br$tiers
  g$flag <- sp$control && br$any_sig
  g$xa <- if (dodged) "compartment" else "group"
  g$x_limits <- levels(d[[g$xa]])
  g$legend_limits <- levels(d$group)
  g$colours <- .ipo_colours(o, levels(d$group))
  g$ylab <- if (is.null(o$y_label)) sp$ylab else o$y_label
  g$title <- if (is.null(o$title)) .ipo_default_title(id, d$compartment, g$flag) else o$title
  .ipo_common(g)
}

.ipo_prepare_fc <- function(id, o, sp) {
  d <- .ipo_frame(id, o, sp, FALSE)
  short <- function(have) {
    if (is.null(.ipo_env$short)) .ipo_env$short <- list(have = have)
    .ipo_skip(id, "needs at least 2 groups with data")
  }
  if (is.character(d)) return(short(character(0)))
  have <- levels(d$group)[sort(unique(as.integer(d$group)))]
  usable <- setdiff(have, .ipo_UNASSIGNED)
  if (length(usable) < 2L) return(short(usable))
  base <- if (!is.null(o$ref_group) && o$ref_group %in% have) o$ref_group else usable[1]
  if (!is.null(o$ref_group) && !identical(base, o$ref_group))
    message("Plot options: ", id, ": baseline group ", .ipo_quote(o$ref_group),
            " has no data in this graph; using ", .ipo_quote(base), " instead")
  s <- .ipo_summary(d, "none")
  base_i <- match(base, levels(d$group))
  key <- function(x) as.integer(x$channel) * (nlevels(d$compartment) + 1L) + as.integer(x$compartment)
  ref <- s[as.integer(s$group) == base_i, , drop = FALSE]
  s$base <- ref$mean[match(key(s), key(ref))]
  s <- s[as.integer(s$group) != base_i & is.finite(s$base), , drop = FALSE]
  if (!nrow(s)) return(.ipo_skip(id, paste0("no compartment has both the baseline group and another group")))
  # A ratio of means is only a fold change when both means are positive
  # (background-corrected means can be <= 0).
  undefined <- !(s$mean > 0 & s$base > 0)
  if (any(undefined)) {
    u <- s[undefined, , drop = FALSE]
    message("Plot options: ", id, ": no fold change where a mean is at or below 0, left out: ",
            paste0(u$group, " (", u$channel, ", ", u$compartment, ")", collapse = "; "))
    s <- s[!undefined, , drop = FALSE]
    if (!nrow(s)) return(.ipo_skip(id, "every mean to compare is at or below 0"))
  }
  s$log2fc <- log2(s$mean / s$base)
  s$channel <- droplevels(s$channel)
  s$compartment <- droplevels(s$compartment)
  shown <- setdiff(levels(d$group), base)
  dodged <- o$layout == "dodged" && nlevels(s$compartment) > 1L
  g <- list(kind = "fc", id = id, o = o, d = s, chart = "bar", dodged = dodged, log_y = FALSE, tiers = 0L,
            base = base, flag = FALSE)
  g$xa <- if (dodged) "compartment" else "group"
  g$x_limits <- if (dodged) levels(s$compartment) else shown
  g$legend_limits <- shown
  g$colours <- .ipo_colours(o, shown)
  g$ylab <- if (is.null(o$y_label)) paste0("log2(group / ", base, ")") else o$y_label
  g$title <- if (is.null(o$title))
    paste0("Cross-group comparison vs ", base, " (log2 fold-change)", .ipo_comp_note(s$compartment)) else o$title
  .ipo_common(g)
}

.ipo_prepare_graph <- function(id) {
  o <- .ipo[[id]]; sp <- .ipo_SPEC[[id]]
  if (is.null(o) || is.null(sp)) return(.ipo_skip(id, "unknown graph"))
  why <- .ipo_unavailable(sp)
  if (!is.null(why)) return(.ipo_skip(id, why))
  if (sp$kind == "fc") .ipo_prepare_fc(id, o, sp) else .ipo_prepare_plain(id, o, sp)
}

.ipo_message <- function(head, body) {
  list(kind = "message", head = head, body = body, w = .ipo_px(1400), h = .ipo_px(900))
}

# NULL means "no plot for this graph": the caller then opens no device.
.ipo_prepare <- function(id) {
  g <- tryCatch(.ipo_prepare_graph(id), error = function(e) {
    message("Plot options: ", id, " failed: ", conditionMessage(e))
    .ipo_message(paste0(id, " could not be drawn"), conditionMessage(e))
  })
  if (!is.null(g)) .ipo_env$n <- .ipo_env$n + 1L
  g
}

.ipo_warning <- function() {
  sh <- .ipo_env$short
  if (is.null(sh)) return(NULL)
  all_groups <- setdiff(.ipo_groups, .ipo_UNASSIGNED)
  missing_groups <- setdiff(all_groups, sh$have)
  body <- if (length(missing_groups) > 0L && length(sh$have) > 0L)
    paste0("Only ", paste(sh$have, collapse = ", "), " has data for this comparison. Missing: ",
           paste(missing_groups, collapse = ", "), ". Check the Intensity picker's image-to-group assignments and the graph's included groups.")
  else
    "Add a second group in the Intensity picker, or include one in the graph's options, to enable cross-group comparison."
  .ipo_env$n <- .ipo_env$n + 1L
  .ipo_message("Fewer than 2 groups have data: comparison plots skipped", body)
}

.ipo_empty <- function() {
  if (.ipo_env$n > 0L) return(NULL)
  body <- if (nrow(.ipo_data) == 0L) "Wire images upstream and Configure intensity."
          else "None of the selected graphs applies to this data. The console lists the reason for each."
  .ipo_message("No intensity plots", body)
}

# -- Building the ggplot (the device is already open) ------------------
.ipo_compose_message <- function(g) {
  ggplot() + theme_void() +
    annotate("text", x = 0.5, y = 0.66, size = 4.6, colour = "#b03030", label = .mpfig_wrap(g$head, 44)) +
    annotate("text", x = 0.5, y = 0.34, size = 3.4, colour = "grey30", label = .mpfig_wrap(g$body, 60)) +
    xlim(0, 1) + ylim(0, 1)
}

.ipo_compose_plain <- function(g) {
  o <- g$o; d <- g$d; s <- g$s; xa <- g$xa; dodged <- g$dodged; chart <- g$chart
  dw <- .ipo_DODGE
  pos <- if (dodged) position_dodge(width = dw) else "identity"
  jit <- if (dodged) position_jitterdodge(jitter.width = 0.12, jitter.height = 0, dodge.width = dw, seed = .ipo_SEED)
         else position_jitter(width = 0.13, height = 0, seed = .ipo_SEED)
  raw_points <- if (o$points)
    geom_point(data = d, aes(x = .data[[xa]], y = y, group = group, fill = group), position = jit,
               size = 0.9, alpha = 0.6, colour = "grey20", stroke = 0, show.legend = FALSE)
  main <- switch(chart,
    bar = geom_col(data = s, aes(x = .data[[xa]], y = mean, fill = group), position = pos,
                   width = if (dodged) dw * 0.9 else 0.7, colour = "grey25", linewidth = 0.3, show.legend = TRUE),
    box = geom_boxplot(data = d, aes(x = .data[[xa]], y = y, fill = group), position = pos,
                       width = if (dodged) dw * 0.8 else 0.6, linewidth = 0.3, show.legend = TRUE,
                       outlier.shape = if (o$points) NA else 19, outlier.size = 0.8),
    violin = geom_violin(data = d, aes(x = .data[[xa]], y = y, fill = group), position = pos,
                         width = if (dodged) dw * 0.9 else 0.8, linewidth = 0.3, scale = "width", show.legend = TRUE),
    mean = geom_point(data = s, aes(x = .data[[xa]], y = mean, colour = group, group = group), position = pos,
                      size = 3, shape = 18, show.legend = TRUE))
  bars <- if (o$error != "none" && chart %in% c("bar", "mean"))
    geom_errorbar(data = s, aes(x = .data[[xa]], ymin = lo, ymax = hi, group = group), position = pos,
                  width = 0.22, linewidth = 0.4)
  p <- ggplot()
  p <- if (chart == "mean") p + raw_points + main + bars else p + main + bars + raw_points
  ex <- expansion(mult = c(if (o$y_zero) 0 else 0.05, 0.08))
  p <- p + if (g$log_y) scale_y_log10(expand = ex)
           else scale_y_continuous(expand = ex, limits = if (o$y_zero && all(d$y >= 0)) c(0, NA) else NULL)
  p <- .ipo_style(p, g)
  if (!is.null(g$br)) {
    size_pt <- getOption("mpfig_base_size", o$font_size)
    args <- list(g$br, label = "label", xmin = "xmin", xmax = "xmax", y.position = "y.position",
                 # With many tiers a bracket's tips would run into the label below.
                 tip.length = if (g$log_y || g$tiers > 8L) 0 else 0.01, bracket.size = 0.3,
                 label.size = size_pt * (if (o$sig == "p") 0.2 else 0.236))
    # Stars sit low in their line box; p-value text needs lifting off the bracket.
    if (o$sig == "p") args$vjust <- -0.35
    p <- tryCatch(p + do.call(ggprism::add_pvalue, args),
                  error = function(e) { message("Plot options: brackets dropped: ", conditionMessage(e)); p })
  }
  p
}

.ipo_compose_fc <- function(g) {
  dw <- .ipo_DODGE
  p <- ggplot(g$d, aes(x = .data[[g$xa]], y = log2fc, fill = group)) +
    geom_col(position = if (g$dodged) position_dodge(width = dw) else "identity",
             width = if (g$dodged) dw * 0.9 else 0.7, colour = "grey25", linewidth = 0.3, show.legend = TRUE) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
    scale_y_continuous(expand = expansion(mult = c(0.08, 0.08)))
  .ipo_style(p, g)
}

.ipo_compose <- function(g) {
  switch(g$kind,
         plain = .ipo_compose_plain(g),
         fc = .ipo_compose_fc(g),
         .ipo_compose_message(g))
}

# Always returns one ggplot that can be printed: a graph that cannot be
# built becomes a message panel, so the plot indices stay aligned.
.ipo_plot <- function(g) {
  fail <- function(e) {
    message("Plot options: ", g$id, " failed: ", conditionMessage(e))
    .ipo_compose_message(list(head = paste0(g$id, " could not be drawn"), body = conditionMessage(e)))
  }
  p <- tryCatch(.ipo_compose(g), error = fail)
  tryCatch({ suppressWarnings(suppressMessages(ggplot_build(p))); p }, error = fail)
}
`;
