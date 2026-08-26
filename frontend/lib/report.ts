import { EvidenceClaim, Observation } from './api';

export type DrawerSelection = { claim: EvidenceClaim; observation: Observation };

function displayClaimText(text: string) {
  return text
    .replace(/\[\s*S\d+\s*\]/g, '')
    .replace(/\(\s*\)|\[\s*\]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

export function cleanReport(report: string, evidence: EvidenceClaim[]) {
  const claimsByText = new Map(
    evidence.map((claim) => [displayClaimText(claim.text), claim.claim_id]),
  );
  const output: string[] = [];
  let claimId: string | null = null;
  let claimSources: string[] = [];
  let pillsPlaced = false;
  for (const line of report.replace(/\n## Sources\s*[\s\S]*$/i, '').trim().split('\n')) {
    if (line.startsWith('## ')) {
      claimId = null;
      claimSources = [];
      pillsPlaced = false;
    }
    if (line.startsWith('### ')) {
      claimId = claimsByText.get(line.slice(4).trim()) ?? null;
      claimSources = [];
      pillsPlaced = false;
    }
    if (/^- (Confidence|Support|Contradiction):/i.test(line)) continue;
    if (/^- Sources:/i.test(line)) {
      claimSources = [...new Set(line.match(/S\d+/g) ?? [])];
      continue;
    }
    if (claimSources.length && line.trim() && !line.startsWith('#')) {
      const citedInline = new Set(line.match(/S\d+/g) ?? []);
      const linkedLine = claimId
        ? line.replace(/\b(S\d+)\b/g, (sourceId) => claimSources.includes(sourceId) ? `[${sourceId}](#evidence-${claimId}-${sourceId})` : sourceId)
        : line;
      const pills = (pillsPlaced ? [] : claimSources.filter((sourceId) => !citedInline.has(sourceId)))
        .map((sourceId) => claimId ? `[${sourceId}](#evidence-${claimId}-${sourceId})` : sourceId)
        .join(' ');
      pillsPlaced = true;
      output.push(pills ? `${linkedLine} ${pills}` : linkedLine);
      continue;
    }
    output.push(line);
  }
  return output.join('\n').trim();
}

export function findCitation(
  evidence: EvidenceClaim[],
  claimId: string,
  sourceId: string,
): DrawerSelection | null {
  const claim = evidence.find((item) => item.claim_id === claimId);
  const observation = claim?.observations.find((item) => item.source_id === sourceId);
  return claim && observation ? { claim, observation } : null;
}
