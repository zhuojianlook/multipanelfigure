/* ──────────────────────────────────────────────────────────
   Image-group helpers — load-time normalization plus the pure
   mutations behind subgroup DIVIDERS inside a group drawer.

   Data model (see ImageGroup / ImageGroupDivider in api/types.ts):
     - group.imageNames is the group's COMPLETE flat membership. A
       divider never "owns" an image away from it — builds that predate
       dividers ignore the `dividers` key and must still find every
       image of the group in imageNames.
     - divider.imageNames is that subgroup's images in display order;
       every one is also in group.imageNames, in at most one divider.
     - Display order: unsectioned images (imageNames minus any divider's
       images, in imageNames order), then each divider: header, images.
     - `dividers` is ABSENT on groups that have none, so a project whose
       user never touched the feature re-saves byte-for-byte the same
       group shape it was loaded with.

   Every mutation here is IMMUTABLE: it never writes to its input and
   returns the very same array reference when nothing changed. That
   lets the store feed it the frozen committed state (immer auto-freezes)
   and skip the unsaved flag + backend sync on a no-op drop.
   ────────────────────────────────────────────────────────── */

import type { ImageGroup, ImageGroupDivider } from "../api/types";

/** One renderable section of a group drawer. */
export interface GroupSection {
  /** null = the group's unsectioned area — always sections[0]. */
  dividerId: string | null;
  /** The divider's name; null for the unsectioned area. */
  name: string | null;
  imageNames: string[];
}

/** Where inside a group a moved image should land. */
export interface SectionTarget {
  /** Divider to drop into; null/undefined (or an unknown id) = unsectioned. */
  dividerId?: string | null;
  /** Insert before this image of the target section; null/undefined, or an
   *  image that isn't in that section, appends to the section's end. */
  beforeImage?: string | null;
  /** The drop was on the section's explicit END zone. Without this, dropping
   *  an image into its OWN section is a no-op (so a stray drop on the section
   *  background doesn't reshuffle it); with it, the image moves to the end. */
  atEnd?: boolean;
}

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** Same scheme createImageGroup has always used — short, unique enough for
 *  ids that only need to be distinct within one project. */
export function newImageGroupId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

/** `base`, or `base-2`, `base-3`… — whichever is first not in `taken`. */
function uniqueId(base: string, taken: Set<string>): string {
  let id = base;
  for (let k = 2; taken.has(id); k++) id = `${base}-${k}`;
  return id;
}

/** Keep only string names that pass `keep`, first occurrence wins. */
function cleanNames(raw: unknown, keep: (n: string) => boolean): string[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const n of raw) {
    if (typeof n !== "string" || seen.has(n) || !keep(n)) continue;
    seen.add(n);
    out.push(n);
  }
  return out;
}

const sameList = (a: string[], b: string[]) =>
  a.length === b.length && a.every((n, i) => n === b[i]);

/* ── Load-time normalization ──────────────────────────── */

/** Turn whatever a .mpf / tab snapshot carried as `image_groups` into a
 *  well-formed ImageGroup[]. Never throws: a corrupted or hand-edited file
 *  must not stop the PROJECT from loading over a cosmetic sidebar feature.
 *
 *  - Not an array → []. Non-object entries are skipped.
 *  - Missing/empty group or divider ids get deterministic ones (so the same
 *    file normalizes the same way twice); duplicates are re-suffixed.
 *  - Image names: strings only, de-duplicated, and (when `validNames` is
 *    given) restricted to images actually in the project.
 *  - Divider images must belong to the group (group.imageNames is the
 *    authority — an older build may have moved the image elsewhere while
 *    ignoring the dividers) and each image lands in its FIRST divider only.
 *  - `dividers` that isn't an array, or normalizes to nothing, is dropped.
 *  - Unknown keys on groups and dividers are carried through untouched so a
 *    file from a NEWER build survives a round-trip through this one.
 *
 *  For a well-formed group without dividers the result is deep-equal to the
 *  `{ ...g, imageNames: g.imageNames.filter(valid) }` every load path used
 *  before dividers existed. */
export function normalizeImageGroups(raw: unknown, validNames?: Set<string>): ImageGroup[] {
  if (!Array.isArray(raw)) return [];
  const out: ImageGroup[] = [];
  const groupIds = new Set<string>();
  raw.forEach((g, gi) => {
    if (!isObj(g)) return;
    const { dividers: rawDividers, ...rest } = g;
    const id = uniqueId(typeof g.id === "string" && g.id ? g.id : `group-${gi + 1}`, groupIds);
    groupIds.add(id);
    const name = typeof g.name === "string" ? g.name : `Group ${gi + 1}`;
    const imageNames = cleanNames(g.imageNames, (n) => !validNames || validNames.has(n));
    const group = { ...rest, id, name, imageNames } as ImageGroup;

    if (Array.isArray(rawDividers)) {
      const member = new Set(imageNames);
      const claimed = new Set<string>();
      const dividerIds = new Set<string>();
      const dividers: ImageGroupDivider[] = [];
      rawDividers.forEach((d, di) => {
        if (!isObj(d)) return;
        const did = uniqueId(typeof d.id === "string" && d.id ? d.id : `${id}-d${di + 1}`, dividerIds);
        dividerIds.add(did);
        const dname = typeof d.name === "string" ? d.name : `Subgroup ${dividers.length + 1}`;
        const names = cleanNames(d.imageNames, (n) => member.has(n) && !claimed.has(n));
        names.forEach((n) => claimed.add(n));
        dividers.push({ ...d, id: did, name: dname, imageNames: names } as ImageGroupDivider);
      });
      if (dividers.length) group.dividers = dividers;
    }
    out.push(group);
  });
  return out;
}

/* ── Sections ─────────────────────────────────────────── */

/** Internal section form: keeps the whole divider object so a rebuild can
 *  spread it and preserve fields this build doesn't know about. */
interface Sec {
  divider: ImageGroupDivider | null;
  imageNames: string[];
}

/** Split a group into [unsectioned, ...dividers]. Defensive even though
 *  state is normalized: a divider name outside the group's membership is
 *  ignored, and an image claimed twice shows in its first divider only. */
function toSecs(group: ImageGroup): Sec[] {
  const member = new Set(group.imageNames);
  const claimed = new Set<string>();
  const dividerSecs: Sec[] = (Array.isArray(group.dividers) ? group.dividers : []).map((d) => {
    const names: string[] = [];
    for (const n of d.imageNames ?? []) {
      if (!member.has(n) || claimed.has(n)) continue;
      claimed.add(n);
      names.push(n);
    }
    return { divider: d, imageNames: names };
  });
  const seen = new Set<string>();
  const unsectioned = group.imageNames.filter((n) => {
    if (claimed.has(n) || seen.has(n)) return false;
    seen.add(n);
    return true;
  });
  return [{ divider: null, imageNames: unsectioned }, ...dividerSecs];
}

/** Rebuild a group from its sections. imageNames is rewritten as the
 *  flattened DISPLAY order, so an older build (which only reads the flat
 *  list) shows the images in the same order this one does. The `dividers`
 *  key is dropped entirely when no divider remains. */
function fromSecs(group: ImageGroup, secs: Sec[]): ImageGroup {
  const next: ImageGroup = { ...group, imageNames: secs.flatMap((s) => s.imageNames) };
  delete next.dividers;
  const dividers = secs
    .filter((s): s is Sec & { divider: ImageGroupDivider } => s.divider !== null)
    .map((s) => ({ ...s.divider, imageNames: s.imageNames }));
  if (dividers.length) next.dividers = dividers;
  return next;
}

/** Ordered sections for rendering a group drawer. sections[0] is always
 *  the unsectioned area (possibly empty); dividers follow in array order. */
export function sectionsOf(group: ImageGroup): GroupSection[] {
  return toSecs(group).map((s) => ({
    dividerId: s.divider ? s.divider.id : null,
    name: s.divider ? s.divider.name : null,
    imageNames: s.imageNames,
  }));
}

/** Replace groups[index] — new array, everything else shared. */
function replaceAt(groups: ImageGroup[], index: number, group: ImageGroup): ImageGroup[] {
  const next = groups.slice();
  next[index] = group;
  return next;
}

/* ── Mutations ────────────────────────────────────────── */

/** Remove an image from one group: its flat membership and any divider.
 *  Returns the same group object when the image wasn't in it. */
export function stripImageFromGroup(group: ImageGroup, imageName: string): ImageGroup {
  const inGroup = group.imageNames.includes(imageName)
    || (group.dividers ?? []).some((d) => (d.imageNames ?? []).includes(imageName));
  if (!inGroup) return group;
  return fromSecs(group, toSecs(group).map((s) => ({
    ...s,
    imageNames: s.imageNames.filter((n) => n !== imageName),
  })));
}

/** Remove an image from every group (image deleted, or sent back to the
 *  timeline). Same array back when no group held it. */
export function stripImageFromGroups(groups: ImageGroup[], imageName: string): ImageGroup[] {
  let changed = false;
  const next = groups.map((g) => {
    const s = stripImageFromGroup(g, imageName);
    if (s !== g) changed = true;
    return s;
  });
  return changed ? next : groups;
}

/** Move an image into a section of a group, keeping the single-location
 *  invariant: it's first stripped from every group (and their dividers).
 *
 *  - No target / dividerId null → the group's unsectioned area, which is
 *    exactly what the old two-argument moveImageToGroup did.
 *  - Dropping into the section the image is ALREADY in with no usable
 *    beforeImage is a no-op (same array back) — re-dropping an image on
 *    its own drawer must not shuffle it to the end.
 *  - beforeImage inserts right before that image within the section.
 *  - Unknown group → no-op. Unknown divider → unsectioned. */
export function moveImageToSection(
  groups: ImageGroup[],
  imageName: string,
  groupId: string,
  target: SectionTarget = {},
): ImageGroup[] {
  const gi = groups.findIndex((g) => g.id === groupId);
  if (gi < 0) return groups;

  const sectionIndex = (secs: Sec[]) => {
    if (!target.dividerId) return 0;
    const k = secs.findIndex((s) => s.divider?.id === target.dividerId);
    return k < 0 ? 0 : k;
  };

  const origSecs = toSecs(groups[gi]);
  const k = sectionIndex(origSecs);
  const before = target.beforeImage ?? null;
  const fromSameSection = origSecs[k].imageNames.includes(imageName);
  if (fromSameSection && (
    (before === null && !target.atEnd)
    || before === imageName
    || (before !== null && !origSecs[k].imageNames.includes(before))
  )) {
    return groups;
  }

  const stripped = stripImageFromGroups(groups, imageName);
  const secs = toSecs(stripped[gi]);
  const tk = sectionIndex(secs);
  const names = secs[tk].imageNames;
  const at = before === null ? -1 : names.indexOf(before);
  const placed = at < 0 ? [...names, imageName] : [...names.slice(0, at), imageName, ...names.slice(at)];
  // A within-section "reorder" that lands the image where it already was
  // (dropped on the tile right after it) changes nothing — report no-op.
  if (fromSameSection && sameList(placed, origSecs[k].imageNames)) return groups;
  secs[tk] = { ...secs[tk], imageNames: placed };
  return replaceAt(stripped, gi, fromSecs(stripped[gi], secs));
}

/** "Subgroup N" with N one past the divider count, bumped past any name
 *  already taken so two dividers never default to the same label. */
export function nextSubgroupName(group: ImageGroup): string {
  const taken = new Set((group.dividers ?? []).map((d) => d.name));
  let n = (group.dividers?.length ?? 0) + 1;
  while (taken.has(`Subgroup ${n}`)) n++;
  return `Subgroup ${n}`;
}

/** Place a new divider in a group.
 *
 *  - Plain: append an EMPTY divider after the last section.
 *  - opts.splitBeforeImage = X: X and every image after it in X's section
 *    move into the new divider, inserted right after X's section — i.e. a
 *    divider is dropped into the gap just left of X. If X isn't in the
 *    group, falls back to appending an empty divider.
 *
 *  Returns the new groups plus the new divider's id (null, with the same
 *  array back, when the group doesn't exist). */
export function addDividerToGroup(
  groups: ImageGroup[],
  groupId: string,
  opts: { name?: string; splitBeforeImage?: string; id?: string } = {},
): { groups: ImageGroup[]; dividerId: string | null } {
  const gi = groups.findIndex((g) => g.id === groupId);
  if (gi < 0) return { groups, dividerId: null };
  const group = groups[gi];
  const secs = toSecs(group);
  const takenIds = new Set((group.dividers ?? []).map((d) => d.id));
  const id = uniqueId(opts.id || newImageGroupId(), takenIds);
  const name = opts.name?.trim() || nextSubgroupName(group);

  let insertAt = secs.length;
  let moved: string[] = [];
  if (opts.splitBeforeImage) {
    const sk = secs.findIndex((s) => s.imageNames.includes(opts.splitBeforeImage!));
    if (sk >= 0) {
      const p = secs[sk].imageNames.indexOf(opts.splitBeforeImage);
      moved = secs[sk].imageNames.slice(p);
      secs[sk] = { ...secs[sk], imageNames: secs[sk].imageNames.slice(0, p) };
      insertAt = sk + 1;
    }
  }
  secs.splice(insertAt, 0, { divider: { id, name, imageNames: [] }, imageNames: moved });
  return { groups: replaceAt(groups, gi, fromSecs(group, secs)), dividerId: id };
}

/** Rename a divider. Same array back if the group/divider is missing or the
 *  name is unchanged. */
export function renameDividerInGroup(
  groups: ImageGroup[],
  groupId: string,
  dividerId: string,
  name: string,
): ImageGroup[] {
  const gi = groups.findIndex((g) => g.id === groupId);
  if (gi < 0) return groups;
  const group = groups[gi];
  const di = (group.dividers ?? []).findIndex((d) => d.id === dividerId);
  if (di < 0 || group.dividers![di].name === name) return groups;
  const dividers = group.dividers!.slice();
  dividers[di] = { ...dividers[di], name };
  return replaceAt(groups, gi, { ...group, dividers });
}

/** Delete a divider. Its images are NOT lost: they merge, in order, onto
 *  the end of the section immediately above it (the previous divider, or
 *  the unsectioned area for the first divider). Same array back if the
 *  group/divider is missing. */
export function deleteDividerFromGroup(
  groups: ImageGroup[],
  groupId: string,
  dividerId: string,
): ImageGroup[] {
  const gi = groups.findIndex((g) => g.id === groupId);
  if (gi < 0) return groups;
  const group = groups[gi];
  const secs = toSecs(group);
  const k = secs.findIndex((s) => s.divider?.id === dividerId);
  if (k <= 0) return groups;
  secs[k - 1] = { ...secs[k - 1], imageNames: [...secs[k - 1].imageNames, ...secs[k].imageNames] };
  secs.splice(k, 1);
  return replaceAt(groups, gi, fromSecs(group, secs));
}
