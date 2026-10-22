# Microwave-assisted digestion, fluorescence staining and image analysis for microplastic quantification: a focused scoping review anchored on two group methods

**Review type:** focused scoping review with a documented, reproducible search (single reviewer)
**Search date:** 2026-10-20
**Databases searched:** OpenAlex, Crossref, Europe PMC, PubMed, arXiv (preprint index)
**Anchor papers:** 2 (the group's own prior art, retrieved by DOI)
**Studies included and extracted:** 287 (abstract available for 257; a quantitative result was extractable for 147)
**Context set reported but not extracted:** 235
**Protocol registration:** none (not registered; a deviation from systematic-review practice, disclosed in §8)
**Working data:** `runs/litreview_v2/` — search manifest, raw responses, citation-chain manifest, screening table, PRISMA counts, extraction table, citation report
**Relationship to v1:** `microplastics_review_v1.md` covered the whole field (identification / shape / polymer type / quantification). This review is **narrower and deeper**: it is scoped to the method stack the group actually works with, and it retrieves the two papers that describe that stack, which v1 did not find.

---

## Abstract

This review synthesises 287 studies retrieved for a specific methodological stack — microwave-assisted or chemical digestion of the matrix, Nile Red fluorescence staining, and image analysis with machine learning — in sediment, soil, fish and other matrices the group works in, anchored on the group's own two published methods. The anchors are the point of the exercise: v1's broad field sweep did not retrieve either paper, because neither title nor abstract uses the generic field terms v1's queries used. Retrieving them by DOI and then expanding them by citation chaining in both directions changes what the review is about, and it exposes the group's own work as a node in a citation network rather than as an isolated result. The findings that matter for the pipeline are these. (i) **The polymer-identification claim is the contested one, and the closest prior art is not what the group may assume.** A Nile Red machine-learning study (2022) reports 95.8 % accuracy for plastic-vs-natural origin and 88.1 % for polymer type, with 92.7 % detection and 80 % identification on spiked environmental samples — but on particles of 50–1200 µm, using RGB rather than a calibrated image stack, and with a known solvatochromic basis that makes PE/PP separation the structurally hard case. (ii) **Detection probability, not model accuracy, is the binding physical limit**: the 2026 automated fluorescence system reports 50 % detection at 40 µm and 90 % at 100 µm, which means a count below ~100 µm is a censored observation, not a measurement. (iii) **Digestion is not a neutral step**: oxidation protocols can alter polymer structure by hydrolysis and can fuse particles at high temperature, so the group's 60 °C ceiling is a substantive finding rather than a convenience. (iv) **Shape taxonomy is now settled and should be adopted, not invented**: a 2025 deep-learning study applies the nine validated categories (fiber, rod, ellipse, oval, sphere, quadrilateral, triangle, free-form, unidentifiable) as reference shapes, giving the group a prior to score against instead of a two-way fibre/fragment split. (v) **Uncertainty reporting remains the field's weak flank**: 22 of 287 extracted studies report uncertainty and 4 are interlaboratory, so a pipeline that reports its own uncertainty would be ahead of the literature rather than merely compliant with it.

---

## 1. Introduction

### 1.1 Background

Two published methods define what the group does. **Perez et al. (2024)**, in *PLOS ONE*, determine PET microplastics in sediment: HNO₃/HCl 3:1, microwave-assisted at 60 °C, 0.45 µm membrane, Nile Red at 10 mg/L, with 97 % recovery and a stated counting limit of particles larger than 0.066 mm. **Ferrada et al. (2024)**, in the *Journal of the Chilean Chemical Society*, determine polyethylene in *Merluccius gayi* fish: HNO₃/H₂O₂ 9:1, microwave-assisted at 60 °C for 1 h, Nile Red at 2 %, with recoveries over 98 % and a demonstration that AI-based selective identification is possible. Between them they cover digestion, staining, imaging and classification — the full stack.

The 2026 FONDECYT proposal built on these two papers but departed from them in the parameters that determine the signal the classifier reads, and overstated what a preliminary dataset of 9,300 particles supports. A new proposal needs a review that is specific to the stack, and that places the group's own work in its context.

### 1.2 Why a focused review, and why anchored

A field-wide review (v1) is the wrong instrument for a proposal that will be judged on a specific method. It spends its synthesis on identification by FTIR, on nanoplastic toxicology and on regulatory context that the proposal will not use. Worse, it did not retrieve the group's own papers: neither title contains the generic terms the field-wide queries used, and both fell outside the screening criteria that admitted 60 studies from 4,175 records. That is a failure of the search, not of the field.

v2 therefore does two things differently. It **retrieves the anchors explicitly by DOI** and admits them by design. And it **expands each anchor by citation chaining in both directions** — the works that cite it, and the works it cites — because a paper that cites or is cited by a group method paper shares its method stack, which keyword search on broad field terms dilutes. Chaining found 33 records, of which 33 survive into the extracted set.

### 1.3 Objectives and review question

**Question.** For the method stack of microwave-assisted or chemical digestion → Nile Red fluorescence staining → image analysis with machine learning, applied to sediment, soil and fish matrices: what does the literature establish about (a) matrix digestion and polymer survival, (b) staining specificity and its confounds, (c) automated detection and classification performance, (d) shape and size reporting, and (e) quantification with its uncertainty — and what does that imply for a group that has already published two methods in this stack and intends to propose a third?

---

## 2. Methods

### 2.1 Eligibility criteria

**Population.** Studies that determine, detect, count, size or classify microplastics in environmental or food matrices, or that develop, validate or harmonise methods for doing so.

**Intervention / exposure.** A sample-treatment method (digestion, density separation, filtration) or a detection/identification method (fluorescence staining, spectroscopy, image analysis with machine learning), or a study of one of these.

**Comparator / reference.** Any other method family, or the same method under different conditions (polymer, matrix, digestion chemistry, instrument).

**Outcomes.** Recovery, selectivity, detection limit, false-positive rate, classification accuracy, shape distribution, size distribution, uncertainty, interlaboratory agreement.

**Study design.** Methodological studies: development, validation, comparison, harmonisation, standardisation, reference-methodology work, or a review of methods. Occurrence studies that report concentrations without contributing methodology are out of scope, as are toxicological, risk-assessment and policy studies.

**Window.** 2015–2026 inclusive, with a documented exception: the two anchor papers and the records reached by citation chaining from them are admitted regardless of year, because chaining is itself the evidence of direct relatedness.

### 2.2 Information sources

OpenAlex, Crossref, Europe PMC (including its preprint index), PubMed and arXiv. All five are open and require no key. **parallel-web was unavailable** (no `PARALLEL_API_KEY` in this environment), so the scripted multi-database search is the whole of the discovery step; that is a limitation and is disclosed in §8.

### 2.3 Search strategy

Three stages, all scripted and all recorded in `sources/manifest.json` and `sources/manifest_chain.json` with the query, the database, the retrieval date, the reported hit count and the status of every call.

**Stage 0 — anchors by DOI.** The two group papers retrieved directly by DOI, with their OpenAlex work IDs recorded.

**Stage 1 — citation chaining, both directions.** For each anchor: the works citing it (OpenAlex `cites:<work_id>`) and the works it cites (OpenAlex `referenced_works`, with Crossref's reference list as a fallback where OpenAlex is thinly indexed — which it is for the Chilean journal, whose record carries an empty reference list). The forward filter requires the OpenAlex work ID rather than the DOI; the first attempt used the DOI and OpenAlex returned HTTP 400, which is recorded in the manifest as a failed call rather than silently dropped.

**Stage 2 — themed queries.** 45 database-query combinations across five themes scoped to the stack: digestion, staining, image analysis, quantification/QA, and matrices. Queries are listed in Appendix A.

**Carryover.** v1's *curated* set of 60 already-screened studies is carried in as a documented source. v1's raw 4,175-record pool is deliberately **not** carried: re-screening an already-screened pool re-admits the occurrence studies v1 already excluded, and the two reviews are meant to be cumulative, not to re-litigate the same corpus.

### 2.4 Selection process

Deduplication by DOI, then by normalised title for records without a DOI. Screening is deterministic and rule-based, with a recorded reason for every exclusion:

1. **Topic** — the title must contain a microplastics term. (Title-scoped on purpose: an earlier version of the screen matched the method stack anywhere in title *and* abstract and admitted 854 records, most of them occurrence studies that mention "sediment" somewhere in the abstract while reporting concentrations. Requiring the signal in the *title* selects studies whose own subject is method work.)
2. **Method focus** — the title must contain a method-development signal; the abstract must confirm one.
3. **Stack and matrix** — the study must touch either a matrix the group works in (sediment, soil, fish/biota, water, and the reference-material and inter-laboratory literature) or a direct method pillar (digestion, staining, image analysis).
4. **Window** — 2015–2026, with the anchor/chain exception.
5. **Tiering** — tier 1 (extracted): at least one direct method pillar (digestion, staining, image analysis) **and** either a group matrix or a standards-setting role (reference methodology, reference material, harmonisation or standardisation, inter-laboratory study, calibration or reproducibility protocol). Tier 2 (context): meets the scope but outside that intersection; reported in the flow, citable, not extracted.

   The standards-setting arm of that rule matters. A reference methodology does not name a matrix, so a matrix-only rule excluded exactly the documents the group's own method must be written against — the Nile Red protocol-harmonisation study and the reference methodology for particle size distribution. Adding the arm moved 42 such studies into the extracted set.

No full text was retrieved or assessed. The consequence is stated in §8 and in the extraction table's `extraction_basis` column, which marks every row as `rule-derived from title+abstract` except the two hand-extracted anchors.

### 2.5 Data extraction

For each included study: citation, DOI, anchor status, retrieval route, stack pillars, evidence type, scope/matrix, method family, the first quantitative result recoverable from the abstract, and four quality flags. The fields are **rule-derived from the title and abstract** by named, auditable regexes, so a reader who disagrees with a cell can see which rule produced it. The two anchors were hand-extracted.

### 2.6 Quality appraisal

No formal risk-of-bias instrument (RoB 2, AMSTAR 2) applies to a methodological scoping review of this kind, so four domain-appropriate flags were recorded per study: reports validation/accuracy/recovery; reports uncertainty or confidence; interlaboratory design; sample size reported. Results in §3.3.

### 2.7 Synthesis

Thematic — by the four stack pillars, not study by study. The two anchors are treated as a focal point rather than as two more rows, because the review's purpose is to locate the group's own work relative to its method stack.

### 2.8 Deviations from protocol

1. **Not registered.** No PROSPERO or OSF registration. Disclosed.
2. **Single reviewer.** No independent duplicate screening or full-text eligibility check. Disclosed, and the reason the extraction table marks every non-anchor row as rule-derived.
3. **No full-text assessment.** Eligibility was judged from title, abstract and metadata.
4. **parallel-web unavailable**, so the scripted search is the entire discovery step.
5. **5 search calls failed**: 4 arXiv calls (HTTP 429 or read timeout, i.e. rate limiting) and the forward chain's first attempt (HTTP 400, because the OpenAlex `cites` filter needs a work ID rather than a DOI). All are recorded in the manifest rather than dropped silently. The arXiv failures affect preprint coverage for T1 and T2 queries; the corrected chaining stage recovered the forward chain for both anchors.

---

## 3. Results

### 3.1 Study selection (PRISMA)

![PRISMA 2020 flow for review v2](prisma_v2.png)

```
Identification
  records from the v2 search (themed + anchors)        4,636
  records carried from v1's curated set                    60
  ------------------------------------------------------
  records total                                          4,696
  duplicates removed                                     1,781
  ------------------------------------------------------
  records screened                                       2,915

Screening
  records excluded                                         2,393
    off-topic: no microplastics term in title+abstract         1,428
    off-topic: no microplastics term in title                      96
    not method-focused (title-scoped signal absent)                510
    not method-focused (no method signal in abstract)               42
    no group matrix and no direct method pillar                   136
    no method-stack pillar present                                120
    outside 2015-2026 window                                       61
  ------------------------------------------------------
  records after screening                                 522

Eligibility
  tier 1 — core, extracted                                 287
  tier 2 — context, reported not extracted                 235
```

**Retrieval route of the extracted set:** 2 by DOI, 33 by citation chain, 252 by themed search. Both anchors are in tier 1.

### 3.2 Characteristics of included studies

| Axis | Distribution |
|---|---|
| **Method family** | fluorescence_staining 108, chemical_digestion 90, image_analysis_ml 63, other 17, microwave_digestion 4, filtration 2, density_separation 1, pyrolysis_gcms 1, spectroscopy_other 1 |
| **Evidence type** | protocol_harmonisation 106, method_comparison 43, method_validation 41, other 40, method_development 38, review 14, application 5 |
| **Scope / matrix** | sediment 94, water 76, fish_biota 70, not stated 26, reference_material 8, food 7, plastic_product 3, air_dust 3 |
| **Year** | the corpus is recent-weighted: 179 of 287 studies are from 2024–2026, and 152 are from 2025–2026 |

**Method family** is dominated by fluorescence staining (108 studies) and chemical digestion (90), with image analysis/ML at 63 and microwave-specific work at 4. That last number is the substantive finding of the whole review: **the microwave-assisted extraction literature is thin.** The group's two papers are the most-cited microwave-digestion method papers in this set.

### 3.3 Quality appraisal

| Flag | Studies (of 287) |
|---|---|
| Reports validation / accuracy / recovery | 183 |
| Reports uncertainty or confidence | 22 |
| Interlaboratory design | 4 |
| Sample size reported | 30 |

**Interpretation.** Validation is common; **uncertainty reporting is not** (22 of 287). Interlaboratory work is rare (4 studies) and, as in v1, carries most of the field's evidence about reliability. The asymmetry — abundant single-laboratory validation, scarce uncertainty and cross-laboratory evidence — is itself a finding, and it is an opening: a pipeline that publishes its own uncertainty would be ahead of this literature rather than merely compliant with it.

---

## 4. Synthesis by theme

### 4.1 Matrix digestion: a method step that changes the analyte

The stack begins with removing the matrix, and the corpus is consistent about how consequential that is. Oxidation protocols "alter the nature of polymers through hydrolysis and often do not lead to a complete matrix removal" (Pfohl et al. 2021), which is the single most important caveat for a proposal that plans to measure polymer recovery. Fenton pretreatment for organic-rich sediment "was found to be efficient in reducing organic matter content, compatible with later FTIR for polymer identification, although it did affect the size of polyethylene and PET" (Vermeiren et al. 2020) — that is, the matrix-removal step changes the *size* of the particles being measured. A systematic evaluation of digestion protocols exists precisely because the choice of digestion chemistry determines both the recovery and the size distribution reported.

This is where the group's own 60 °C ceiling becomes a **substantive methodological claim rather than a convenience**: the PLOS ONE paper reports that above 120 °C PET recovery fell because particles fused, and at 60 °C recovery was 97 %. The 2024 J. Chil. Chem. Soc. paper extends the same 60 °C logic to an acid/oxidant route in fish tissue with recoveries over 98 %, and additionally shows that lyophilization — a step common in the literature — is unnecessary and itself causes losses (≈65 % recovery).

**Implication for the pipeline.** Recovery and polymer survival are not a validation afterthought; they are parameters that must be fixed, reported and defended. A proposal that says "microwave digestion at ≤80 °C" (the 2026 draft's wording) is claiming a protocol *between* two published ones. Either restate one published protocol precisely, or make the optimisation of the conditions an explicit objective with pre-declared acceptance criteria. The 2026 draft did neither, and in doing so cited each published paper for a method the other describes.

### 4.2 Staining: the method is a screen, and its confounds are known

Nile Red's mechanism is stated plainly in the corpus: "the solvatochromic nature of Nile Red also offers the possibility of plastic categorisation based on surface polarity characteristics of identified particles" (Maes et al. 2017, *Scientific Reports*). That single sentence is the whole polymer-identification question in one line — the dye reports **polarity**, so polymer assignment is an inference from colour, and two polymers of similar polarity are structurally confusable. It also explains why the group's own 2024 paper reports PE as bright yellow while its 2024 PLOS ONE paper could not make colour do the work.

The corpus also carries the two things a proposal must handle. **False positives are pervasive**: 47 of the 287 extracted studies mention interference, false positives or organic-matter effects, and the standard remedy in the literature is a second stain or an enzymatic step to remove biological material. And **staining conditions are not standardised** — a 2025 study in the corpus is explicitly about harmonising Nile Red staining protocols (Ho and Masura 2025), which is a direct indication that the field recognises its protocols are not comparable across laboratories. The remedy is not merely aspirational: the corpus also carries a **reference methodology for microplastic particle size distribution analysis** (Richter et al. 2022), which is exactly the kind of document the group's own method should be written against.

**Implication for the pipeline.** A count from Nile Red-stained, fluorescence-imaged particles is a count of *stained particles* until a confounder control says otherwise. Blanks, the substrate that produces false positives, and a second stain are not optional extras; they are the difference between a count and a measurement. The 2026 draft proposed "rinsed with distilled water" as its contamination control, which is not a protocol.

### 4.3 Automated detection and classification: the closest prior art, and what it actually supports

This is the theme where anchoring changes the answer, because the closest prior art to the group's own approach is a paper a field-wide keyword search dilutes but a citation chain surfaces.

**The closest prior art.** Meyers et al. (2022), in *Science of the Total Environment*, combine Nile Red staining with RGB data from photographs to train two supervised models: a **Plastic Detection Model** and a **Polymer Identification Model**. It reports 95.8 % accuracy for plastic-versus-natural origin and 88.1 % for polymer type, and 92.7 % detection and 80 % identification on spiked environmental samples after laboratory processing. The pipeline is: stain, photograph, extract RGB, classify. That is *architecturally the same idea* as the group's descriptor-based polymer identification model.

**But read the conditions.** The particles are 50–1200 µm. The input is an RGB photograph, not a calibrated image stack with a recorded µm/px scale. The particles are spiked standards processed in the laboratory, not environmental samples with a confounder load. And the polymer question is exactly the one the dye's solvatochromism makes hard.

**The physical limit.** Tanoiri et al. (2026) are the reference point in this corpus for what an automated fluorescence system can actually do: **detection probability 50 % at 40 µm and 90 % at 100 µm**. Those numbers are the most consequential in this review, because they are *not* model properties. They are properties of the optics and the dye, and they bound what any classifier can achieve. A model reporting 95 % accuracy on particles it can only see 50 % of the time is reporting a conditional accuracy over a detection-limited subset, and the distinction has to be made explicit in any proposal or paper.

**The group's own position.** The measured per-class recall in the group's pipeline is PET 0.98, PP 0.70, LDPE 0.54. The 2022 study's 88.1 % polymer accuracy is *higher* than the group's PP and LDPE figures. That is not a reproach to the group's pipeline — it is a different setting, a different size range and a different label space — but it does mean the proposal cannot claim novelty on "automated polymer identification by Nile Red", because it was demonstrated in 2022. The defensible novelty is elsewhere, and §6 says where.

### 4.4 Shape and size: a taxonomy exists, and it should be adopted

The corpus contains a 2025 deep-learning study (Liu et al. 2025) that applies the nine validated shape categories — fiber, rod, ellipse, oval, sphere, quadrilateral, triangle, free-form, unidentifiable — as reference shapes for hyperspectral-imaged microplastics. Those are Liu et al.'s categories, and their presence here means the group can score against a published, expert-validated taxonomy rather than inventing a two-way fibre/fragment split.

The cost of adopting them should be stated plainly: the group's current `morfotipo` is a two-way split with a 1.1 % fibre rate over 6,638 measured particles. Adopting the nine categories **will change the measured shape distribution**, so it is a results-affecting decision, not a cosmetic one.

For size, the corpus is consistent that maximum Feret's diameter is the reporting quantity, and the 2026 "NRmachine" study's size-dependent detection curve gives the complementary half: what you *see* depends on the size, so a size distribution and a detection probability have to be published together or the count is uninterpretable.

### 4.5 Quantification and uncertainty: the field's weak flank, and the group's opportunity

Recovery is reported often (183 of 287 studies report validation, accuracy or recovery) and is usually good — the corpus contains 93 ± 10 % recovery from mussel tissue by NaOH and enzyme digestion (Catarino et al. 2016), 75 % and 77 % recovery for pretreatment and infrared imaging in a validated imaging-FTIR method (Uurasjärvi et al. 2021), and > 98 % for the group's own microwave-assisted oxidant route. **Uncertainty is reported rarely** (22 of 287). Detection limits appear in only a handful of studies, and only 4 are interlaboratory.

**This asymmetry is the opportunity, and it is a concrete one.** The v1 review established that no fully validated quantification method exists in the field (Vitali et al. 2023) and that interlaboratory agreement on polymer type ranges from 53 % to 100 % (van Mourik et al. 2021). v2 shows the same pattern inside the group's own niche, and shows that it is *not* because the field is full of careless work — it is full of careful single-laboratory recovery studies and almost no uncertainty.

---

## 5. Standards and regulatory context

The relevant standards instruments are the ISO 24187:2023 series (microplastics in the environment) and the VAMAS prestandardisation platform, alongside national guidance documents in the corpus (an Arctic monitoring guidelines document appears among the extracted set). The corpus's own harmonisation literature — the 2025 Nile Red protocol harmonisation study, the inter-laboratory extraction comparisons, the reference-material work — is where a proposal should look for the *shape* of a validation package, because the group is proposing to build one.

For the specific claim the group's 2024 papers support — a microwave-assisted, Nile Red, consumer-camera method — the relevant precedent in the corpus is Richter et al. (2022)'s **reference methodology for microplastic particle size distribution analysis**, and the harmonisation effort of Ho and Masura (2025) for Nile Red staining protocols.

---

## 6. What this means for the next proposal

**The novelty claim must move.** "Automated polymer identification by Nile Red" was demonstrated in 2022 at 88.1 % accuracy. The defensible contributions are the ones the corpus shows nobody has done: a **detection-probability-vs-size model** for the group's own optical stack, so that counts are censored observations with a stated detection limit; **recovery and polymer survival reported per polymer per matrix per digestion chemistry** rather than as a single headline percentage; **the unassignable fraction reported as a result**, which follows directly from the dye's solvatochromism; and **the synthetic-to-real-image performance drop** for a learned detector, which no study in this corpus reports.

**The preliminary data must be real.** The corpus contains field-plate-independent method-development work, but no study trains only on synthetic images and reports the drop on real filters. The group's M007 pipeline is currently a synthetic-corpus comparison whose committed verdict is an honest REVIEW with 11 of 15 gates failing. That is a legitimate *methods* contribution — pre-declared gates, honest negative results — and it is the kind of thing this field does not have. But it is not a performance claim, and a proposal should present it as the former.

**The polymer scope must be decided.** The proposal's target of PE, PET, PP, PVC, PA is not the code's PET, PP, LDPE, and not either paper's scope (PET in sediment; PE in fish). The literature says the dye gives a polarity class, that polymer identity is the confusable case, and that identity belongs to µFTIR or Raman. The honest and achievable claim is *counting and sizing with a polarity-based classification and explicit abstention, with a spectroscopy-confirmed subset*.

**The method must be stated as one of the two published protocols, not as a blend.** HNO₃/HCl 3:1 at 60 °C with Nile Red 10 mg/L, or HNO₃/H₂O₂ 9:1 at 60 °C for 1 h with Nile Red 2 % — each with its published recovery, its published matrix and its published caveats. A third protocol should be a stated objective with pre-declared acceptance criteria, not an undeclared midpoint.

---

## 7. Evidence gaps

1. **Detection probability is not modelled.** The corpus contains one study that reports it (50 % at 40 µm, 90 % at 100 µm). No study in the extracted set propagates it into a count's uncertainty.
2. **No synthetic-to-real-image drop.** No extracted study trains only on synthetic images and reports the performance drop on real filters.
3. **The microwave-assisted extraction literature is thin.** 4 of 287 extracted studies are microwave-specific, and the two most-cited of them are the group's own.
4. **Uncertainty is reported in 22 of 287 studies**, and interlaboratory design in 4.
5. **Staining protocols are not harmonised** for Nile Red in the matrices the group works in, and a harmonisation study appeared only in 2025.
6. **Size-dependent and matrix-dependent recovery are not reported together**, so a recovery figure cannot be transferred between matrices.

---

## 8. Limitations

1. **No full-text assessment.** Eligibility was judged from title, abstract and metadata; 0 full-text reports were retrieved. The extraction table's `extraction_basis` column marks every non-anchor row `rule-derived from title+abstract`. Any statement about a specific study's *method detail* beyond what its abstract states is therefore not supported by this review.
2. **Single reviewer.** No duplicate screening, no independent full-text eligibility check, and no inter-reviewer disagreement to report. v1 had the same limitation; v2 adds that the screening rules were written, applied and revised by the same person, so the eligibility boundary is less independent than a two-reviewer screen would be.
3. **parallel-web unavailable**, so scripted search is the entire discovery step and no supplementary grey-literature discovery was performed.
4. **Five arXiv calls failed** with HTTP 429 / timeouts. Preprint coverage for the T1 and T2 queries is therefore incomplete.
5. **The forward citation chain was initially broken** (HTTP 400: the filter needs an OpenAlex work ID, not a DOI) and was recovered in a separate recorded stage. The final chain totals are 9 forward + 23 backward for A1 and 3 forward + 0 backward for A2; the second anchor has a sparse citation network, partly because the journal is thinly indexed in OpenAlex.
6. **Tier 2 (235 studies) was not extracted**, by design. It is reported in the flow and citable, but nothing in §4 rests on it.
7. **Extraction of the two anchors was by hand from the published PDFs** held in the project's `published-papers/` folder; the corpus's abstracts were not the source for those two rows.
8. **Citation verification checks DOI registration and metadata, not whether a source supports a claim.** Every claim attributed to a specific study in §4 was taken from that study's own abstract text as retrieved, and any claim to be used in a Methods section must be re-checked against the full text.

---

## 9. Conclusion

Scoping a review to a method stack, and anchoring it on the group's own two papers, changes the conclusions in three ways that a field-wide sweep missed. First, **the polymer-identification novelty the 2026 draft assumed does not exist** — a 2022 study built the same stain-and-classify architecture and reported higher per-class accuracy than the group's own pipeline achieves for PP and LDPE. Second, **the real ceiling is detection probability, not classifier accuracy**: the 50 %-at-40 µm figure is a property of optics and dye, and no model can exceed it, which reframes what a preliminary-data section should claim. Third, **the field's genuine weakness is uncertainty and harmonisation, not capability** — 22 of 287 studies report uncertainty and 4 are interlaboratory, in a niche where recovery is routinely 90 %+ and the group's own microwave routes exceed 97 %.

That combination points to a specific, defensible next proposal: take the two validated microwave protocols, extend them to a matrix suite with per-polymer recovery and survival, add a detection-probability model so counts become censored observations with stated limits, report the polarity-based classification with its abstention fraction as a result, and confirm a subset spectroscopically. The method is published, the recovery is high, the optics are cheap, and the quantification — the part nobody has done — is exactly where the group's software platform and pre-registered comparison methodology are the contribution.

---

## References

- (authors unavailable) (2020). *Abundance and Distribution of Microplastics in the Water and Riverbank Sediment in Malaysia – A Review*. Biointerface Research in Applied Chemistry. https://doi.org/10.33263/briac114.1170011712
- (authors unavailable) (2025). *DETECTION AND CLASSIFICATION OF MICROPLASTICS IN WATER SOURCE USING SVM*. IARJSET. https://doi.org/10.17148/iarjset.2025.12125
- (authors unavailable) (n.d.). *(title unavailable)*. . https://doi.org/
- (authors unavailable) (n.d.). *(title unavailable)*. . https://doi.org/
- (authors unavailable) (n.d.). *(title unavailable)*. . https://doi.org/
- (authors unavailable) (n.d.). *(title unavailable)*. . https://doi.org/
- (authors unavailable) (n.d.). *(title unavailable)*. . https://doi.org/
- (authors unavailable) (n.d.). *(title unavailable)*. . https://doi.org/
- (authors unavailable) (n.d.). *(title unavailable)*. . https://doi.org/
- (authors unavailable) (n.d.). *(title unavailable)*. . https://doi.org/
- (authors unavailable) (n.d.). *(title unavailable)*. . https://doi.org/
- (authors unavailable) (n.d.). *(title unavailable)*. . https://doi.org/
- (authors unavailable) (n.d.). *(title unavailable)*. . https://doi.org/
- (authors unavailable) (n.d.). *(title unavailable)*. . https://doi.org/
- A.Manoj Prabaharan (2026). *AI-Driven Raman Spectroscopy and Quantum Dot Probes with Federated Learning for Micro-Nanoplastic Detection and Removal in Livestock and Aquaculture Feeds*. Preprints.org. https://doi.org/10.20944/preprints202601.1512.v1
- Adil Bakir et al. (2023). *A spatial and temporal assessment of microplastics in seafloor sediments: A case study for the UK*. Frontiers in Marine Science. https://doi.org/10.3389/fmars.2022.1093815
- Adrián López-Rosales et al. (2022). *A reliable method for the isolation and characterization of microplastics in fish gastrointestinal tracts using an infrared tunable quantum cascade laser system*. Marine Pollution Bulletin. https://doi.org/10.1016/j.marpolbul.2022.113591
- Ahmad Cahyono Adi & Wahyono (2026). *Adaptive Multi-Scale Gaussian-Laplacian Pyramid with Gabor Filtering for Microplastics Detection*. Engineering Technology & Applied Science Research. https://doi.org/10.48084/etasr.15206
- Ahmad Ghassan Shabib et al. (2025). *Design, fabrication, and application of electrochemical sensors for microplastic detection: a state-of-the-art review and future perspectives*. Environmental Sciences Europe. https://doi.org/10.1186/s12302-025-01138-1
- Ajinkya Nene et al. (2025). *Recent advances and future technologies in nano-microplastics detection*. Environmental Sciences Europe. https://doi.org/10.1186/s12302-024-01044-y
- Akhil Gupta & Pratik Kumar (2025). *Technologies for Polymer Identification and Monitoring of Microplastics Distribution*. . https://doi.org/10.1002/9781394251100.ch5
- Aleksandra Dmitrowicz, Mariusz Sacharczuk & Dominik S. Skiba (2026). *Micro- and nanoplastics in human biological materials: a systematic review of detection methods and methodological challenges*. Archives of Toxicology. https://doi.org/10.1007/s00204-026-04392-1
- Alena Sakovich et al. (2022). *Keeping a sharp Weka eye on tiny plastics – the usability and perspective of Fiji image analysis tool for microplastics identification in environmental samples*. Zenodo (CERN European Organization for Nuclear Research). https://doi.org/10.5281/zenodo.7217190
- Alessandro Rimmaudo et al. (2026). *Quantification and Characterisation of Microplastics in Organic Waste-Derived Soil Amendments*. Water Air & Soil Pollution. https://doi.org/10.1007/s11270-026-09145-5
- Alexandra M. Gulizia et al. (2022). *Evaluating the Effect of Chemical Digestion Treatments on Polystyrene Microplastics: Recommended Updates to Chemical Digestion Protocols*. Macromolecular Chemistry and Physics. https://doi.org/10.1002/macp.202100485
- Ali Karami et al. (2016). *A high-performance protocol for extraction of microplastics in fish*. The Science of The Total Environment. https://doi.org/10.1016/j.scitotenv.2016.10.213
- Alissa H. Tophinke et al. (2022). *Systematic development of extraction methods for quantitative microplastics analysis in soils using metal-doped plastics*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2022.119933
- Amardeep S. G. Singh, Prof. Charles Jia & Donald W. Kirk (2023). *Implementing Edge Based Object Detection For Microplastic Debris*. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2307.16289
- Ana Belén Lozano Avilés, Ginés Morales Méndez & Francisco del Cerro Velázquez (2025). *Evaluation of the Presence of Microplastics in Wastewater Treatment Plants: Development and Verification of Strategies for Their Quantification and Removal in Aqueous Streams*. Sustainability. https://doi.org/10.3390/su17219470
- Ana Carolina Torregroza-Espinosa et al. (2025). *Application of Remote Sensing for the Detection and Monitoring of Microplastics in the Coastal Zone of the Colombian Caribbean*. Microplastics. https://doi.org/10.3390/microplastics4040077
- Ana Isabel Catarino et al. (2016). *Development and optimization of a standard method for extraction of microplastics in mussels by enzyme digestion of soft tissues*. Environmental Toxicology and Chemistry. https://doi.org/10.1002/etc.3608
- Anderson Targino da Silva Ferreira et al. (2025). *Microplastic Deposit Predictions on Sandy Beaches by Geotechnologies and Machine Learning Models*. Coasts. https://doi.org/10.3390/coasts5010004
- Andrea Dali et al. (2026). *Advances in the sample pretreatment for determination of microplastics in riverbed sediments by Raman spectroscopy*. Environmental Science and Pollution Research. https://doi.org/10.1007/s11356-026-38231-8
- Andres Sanchez Garcia et al. (2025). *Microplastics in Stormwater: Sampling and Methodology Challenges*. Toxics. https://doi.org/10.3390/toxics13060502
- Anna T. Kukkola et al. (2023). *Easy and accessible way to calibrate a fluorescence microscope and to create a microplastic identification key*. MethodsX. https://doi.org/10.1016/j.mex.2023.102053
- Anna T. Kukkola et al. (2025). *Fluorescence-Guided Raman Spectroscopy with an Integrated Adapter for Faster and Cost-Effective Microplastic Detection*. Analytical Chemistry. https://doi.org/10.1021/acs.analchem.5c04637
- Aoxiang Xu et al. (2026). *Classification, Quantification, and Fluorescence Behavior Analysis of Microplastics via Nile Red Staining: Application to Detection Emission in Takeaway Food Container*. ACS Omega. https://doi.org/10.1021/acsomega.6c00263
- Aran Akbari et al. (2026). *Assessment, characterization, and quantification of microplastics from urban rivers (Zarjub and Goharrood), Rasht, Iran*. Scientific Reports. https://doi.org/10.1038/s41598-026-72655-z
- Arianna Bellasi et al. (2021). *The extraction of microplastics from sediments: An overview of existing methods and the proposal of a new and green alternative*. Chemosphere. https://doi.org/10.1016/j.chemosphere.2021.130357
- Asim Laeeq Khan & Asad Ali Zaidi (2025). *Separation and Detection of Microplastics in Human Exposure Pathways: Challenges, Analytical Techniques, and Emerging Solutions*. Journal of Xenobiotics. https://doi.org/10.3390/jox15050154
- Aswir Abd Rashed, Nurliayana Ibrahim & Mohammad Adi Mohammad Fadzil (2026). *Leveraging the Advanced Capability of Laser Direct Infrared Imaging (LDIR): A Preliminary Analysis of Microplastics in Edible Tissue of Malaysian Fish*. Microplastics. https://doi.org/10.3390/microplastics5020089
- Aujeeta Shehrin Razzaque & Assefa Mekonnen Melesse (2025). *Global Insights into Micro- and Nanoplastic Pollution in Surface Water: A Review*. Hydrology. https://doi.org/10.3390/hydrology12100265
- Azam Ali & Muhammad Zaman Khan (2026). *Textile Microplastics in Wastewater: A Critical Review of Removal and Carbonization Technologies*. C – Journal of Carbon Research. https://doi.org/10.3390/c12010024
- Babita Thakur et al. (2023). *Identification and characterization of extracted microplastics from agricultural soil near industrial area: FTIR and X‐ray diffraction method*. Environmental Quality Management. https://doi.org/10.1002/tqem.22035
- Bashir Ahmed, Khushi Ali & Syed Ali Hassan (2026). *Advanced Remediation of Heavy Metals, Microplastics, and Polycyclic Aromatic Hydrocarbons (PAHs) in Co-contaminated Agricultural Soils: A Review*. International Journal of Agriculture and Biology. https://doi.org/10.17957/ijab/15.2520
- Benedetto Ardini et al. (2025). *Fast Detection and Classification of Microplastics by a Wide-Field Fourier Transform Raman Microscope*. Environmental Science & Technology. https://doi.org/10.1021/acs.est.5c00165
- Bhairav Prasad et al. (2026). *Integrated approaches for microplastic detection, risk assessment and sustainable biotechnological remediation*. Discover Environment. https://doi.org/10.1007/s44274-026-00959-y
- Bhanuprasad Telu et al. (2025). *The Development of Sensors for Microplastic Detection Using Artificial Intelligence*. International Journal of Artificial Intelligence. https://doi.org/10.36079/lamintang.ijai-01202.934
- Brian Nguyen et al. (2019). *Separation and Analysis of Microplastics and Nanoplastics in Complex Environmental Samples*. Accounts of Chemical Research. https://doi.org/10.1021/acs.accounts.8b00602
- Brian R. Coleman (2024). *An introduction to machine learning tools for the analysis of microplastics in complex matrices*. Environmental Science Processes & Impacts. https://doi.org/10.1039/d4em00605d
- Bruna  Nascimento de Vasconcel Dias et al. (2026). *Automated classification of microplastics from aquatic environments using image analysis and Deep Learning*. SSRN Electronic Journal. https://doi.org/10.2139/ssrn.7069672
- C.M. Penso et al. (2026). *Microplastic Identification Methods for Microfluidic Applications: Towards Rapid Detection in Aquatic Environments*. Polymers. https://doi.org/10.3390/polym18151847
- Carlo Giacomo Avio, Stefania Gorbi & Francesco Regoli (2015). *Experimental development of a new protocol for extraction and characterization of microplastics in fish tissues: First observations in commercial species from Adriatic Sea*. Marine Environmental Research. https://doi.org/10.1016/j.marenvres.2015.06.014
- Carmine Massarelli, Claudia Campanale & Vito Felice Uricchio (2021). *A Handy Open-Source Application Based on Computer Vision and Machine Learning Algorithms to Count and Classify Microplastics*. Water. https://doi.org/10.3390/w13152104
- Changwei Zhu (2026). *The Development and Prospect of Microplastic Detection and Analysis Technology*. MATEC Web of Conferences. https://doi.org/10.1051/matecconf/202642003004
- Chaochao Ma et al. (2026). *Interfacial Regulation–Driven Dual‐Enrichment SERS Coupled With Deep Learning Enable Ultrasensitive and In Situ Identification of Microplastics in Natural Waters*. Small. https://doi.org/10.1002/smll.74231
- Charles E. Bardawil et al. (2025). *Making Fluorescent Nylon, Polypropylene, and Polystyrene Microplastics for In Vivo and In Vitro Imaging*. Microplastics. https://doi.org/10.3390/microplastics4040084
- Chattanchal Ashwathi et al. (2025). *Microplastic and trace element contamination in coastal agricultural soils of southern India: a comparative risk assessment of mulched and unmulched fields*. Environmental Geochemistry and Health. https://doi.org/10.1007/s10653-025-02746-9
- Chiara Capolungo et al. (2021). *Photoluminescence‐Based Techniques for the Detection of Micro‐ and Nanoplastics*. Chemistry - A European Journal. https://doi.org/10.1002/chem.202102692
- Chingakham Chinglenthoiba et al. (2025). *Quantification and identification of microplastic in sediment and surface water from selected areas in the Brunei ecosystem*. Water Emerging Contaminants & Nanoplastics. https://doi.org/10.20517/wecn.2025.04
- Chloe Way et al. (2022). *Assessing the effectiveness of microplastic extraction methods on fishmeal with different properties*. Analytical Methods. https://doi.org/10.1039/d1ay02154k
- Christel S. Hassler et al. (2026). *Revisiting microplastic pollution: A novel method for detecting small-sized microplastics in natural waters*. npj Clean Water. https://doi.org/10.1038/s41545-026-00612-4
- Christina J. Thiele, Malcolm D. Hudson & Andrea E. Russell (2019). *Evaluation of existing methods to extract microplastics from bivalve tissue: Adapted KOH digestion protocol improves filtration at single-digit pore size*. Marine Pollution Bulletin. https://doi.org/10.1016/j.marpolbul.2019.03.003
- Cihang Yang et al. (2024). *Machine learning driven methodology for enhanced nylon microplastic detection and characterization*. Scientific Reports. https://doi.org/10.1038/s41598-024-54003-1
- Cinzia Bettiol et al. (2008). *Evaluation of microwave-assisted acid extraction procedures for the determination of metal content and potential bioavailability in sediments*. Applied Geochemistry. https://doi.org/10.1016/j.apgeochem.2007.11.008
- Clarita Ferrada et al. (2024). *EVALUATION OF MICROWAVE-ASSISTED ACID/OXIDANT DIGESTION METHOD FOR THE DETECTION OF POLYETHYLENE MICROPLASTICS IN Merluccius Gayi FISH BY NILE RED FLUORESCENT STAINING AND IMAGE ANALYSIS*. Journal of the Chilean Chemical Society. https://doi.org/10.4067/s0717-97072024000106082
- Claudia Araceli Dávila‐Camacho et al. (2026). *Physicochemical Characteristics of Microplastics Present in Sediments of the Veracruz Reef System National Park Using Nile Red and SEM/EDS Methods*. Toxics. https://doi.org/10.3390/toxics14070610
- Claudia Lorenz et al. (2019). *Spatial distribution of microplastics in sediments and surface waters of the southern North Sea*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2019.06.093
- Clementina Vitali et al. (2022). *Quantitative image analysis of microplastics in bottled water following Nile Red staining and fluorescence microscopy*. ChemRxiv. https://doi.org/10.26434/chemrxiv-2022-cr5ws
- Clementina Vitali et al. (2023). *Quantitative image analysis of microplastics in bottled water using artificial intelligence*. Talanta. https://doi.org/10.1016/j.talanta.2023.124965
- Daniela Escalona Thomas et al. (2020). *Sample Preparation Techniques for the Analysis of Microplastics in Soil—A Review*. Sustainability. https://doi.org/10.3390/su12219074
- Daniela Honorato‐Zimmer et al. (2021). *Mountain streams flushing litter to the sea – Andean rivers as conduits for plastic pollution*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2021.118166
- Darl-leen A. Gacilos et al. (2024). *Assessment of Microplastics in Fish GI Tracts and Shallow Water Sediments in Pugad Baboy Mangrove Area, Kawit, Cavite Using Fourier Transform Infrared-Attenuated Total Reflectance Spectroscopy*. PUP Journal of Science and Technology. https://doi.org/10.70922/094kv205
- David B. Olawade et al. (2026). *Artificial intelligence for modeling and reducing microplastic in marine environments: A review of current evidence*. Marine Pollution Bulletin. https://doi.org/10.1016/j.marpolbul.2026.119379
- Delphine Ciréderf Boulant et al. (2025). *Method Validation: Extraction of Microplastics from Organic Fertilisers*. Environments. https://doi.org/10.3390/environments12050143
- Denise Margarita Rivera-Rivera et al. (2025). *Exploring Innovative Approaches for the Analysis of Micro- and Nanoplastics: Breakthroughs in (Bio)Sensing Techniques*. Biosensors. https://doi.org/10.3390/bios15010044
- Derek Ho & Julie Masura (2025). *Dyeing to Know: Harmonizing Nile Red Staining Protocols for Microplastic Identification*. Colorants. https://doi.org/10.26434/chemrxiv-2024-wbf7n-v3
- Derek Ho & Julie Masura (2025). *Dyeing to Know: Harmonizing Nile Red Staining Protocols for Microplastic Identification*. Colorants. https://doi.org/10.3390/colorants4020020
- Devjani Chakraborty et al. (2026). *Microplastic pollution across Bangladesh: A review of current evidence and research gaps*. Next Sustainability. https://doi.org/10.1016/j.nxsust.2026.100527
- Dilara Hatinoglu et al. (2024). *Enhancing Microplastics Recovery from Complex Sludge Samples Using COD-Guided Pretreatment*. Water Air & Soil Pollution. https://doi.org/10.1007/s11270-024-07102-8
- Dipayan Das et al. (2026). *Rapid and High‐Resolution Microplastic Detection Technologies: Innovations Toward Sustainable Environmental Monitoring*. Environmental Quality Management. https://doi.org/10.1002/tqem.70475
- Dominika Uchmanowicz et al. (2025). *Microplastics in Airborne Particulate Matter: A Comprehensive Review of Separation Techniques, In Vitro Toxicity and Health Impacts*. International Journal of Molecular Sciences. https://doi.org/10.3390/ijms262110332
- Dr. S.K. Mishra (2022). *Methods for sampling and detection of microplastics in water and sediment: A critical review*. Zenodo (CERN European Organization for Nuclear Research). https://doi.org/10.5281/zenodo.8241054
- Dr. Tasneem K. H. Khan (2026). *Microplastic Contamination in Freshwater Ecosystems: Sources, Detection Strategies and Mitigation Technologies*. International Journal of Integrative Studies (IJIS). https://doi.org/10.63856/egb2j861
- Dr.T. Vengatesh et al. (2025). *A Machine Learning Approach To Microplastic Detection And Quantification In Aquatic Environments*. International Journal of Environmental Sciences. https://doi.org/10.64252/81s3c519
- Edward P. C. Lai et al. (2026). *Development of a Dye-Binding Method for Nanoplastics Detection in Snow Samples Using Capillary Electrophoresis with Laser-Induced Fluorescence Detection*. ACS Omega. https://doi.org/10.1021/acsomega.5c13510
- Eleonora Nicolai et al. (2022). *A New Optical Method for Quantitative Detection of Microplastics in Water Based on Real-Time Fluorescence Analysis*. Water. https://doi.org/10.3390/w14203235
- Eliasz Dzierżyński et al. (2024). *Microplastics in the Human Body: Exposure, Detection, and Risk of Carcinogenesis: A State-of-the-Art Review*. Cancers. https://doi.org/10.3390/cancers16213703
- Emilia Uurasjärvi et al. (2021). *Validation of an imaging FTIR spectroscopic method for analyzing microplastics ingestion by Finnish lake fish (Perca fluviatilis and Coregonus albula)*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2021.117780
- Emilie Kallenbach et al. (2021). *Chitinase digestion for the analysis of microplastics in chitinaceous organisms using the terrestrial isopod Oniscus asellus L. as a model organism*. The Science of The Total Environment. https://doi.org/10.1016/j.scitotenv.2021.147455
- Emily Curren et al. (2021). *Marine microplastics in the ASEAN region: A review of the current state of knowledge*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2021.117776
- Erin E. Tuttle (2023). *Photic zone plastic: isolation of microplastics in environmental samples and improved understanding of their fate in sunlit water*. . https://doi.org/10.17760/d20560793
- Fabio Corradini (2021). *Development and application of tests for microplastic detection in soil*. . https://doi.org/10.18174/535250
- Farhatun Najat Maluin et al. (2026). *Microplastics as trojan horse contaminant vectors of heavy metals and pesticides in the environment*. Microchemical Journal. https://doi.org/10.1016/j.microc.2026.119697
- Fatmawati Marasabessy et al. (2025). *Identification and Quantification of Microplastics in Mackerel (Rastrelliger sp) at the Tual City Fish Market*. Agrikan Jurnal Agribisnis Perikanan. https://doi.org/10.52046/agrikan.v18i2.2653
- Felix Pfeiffer & Elke Kerstin Fischer (2020). *Various Digestion Protocols Within Microplastic Sample Processing—Evaluating the Resistance of Different Synthetic Polymers and the Efficiency of Biogenic Organic Matter Destruction*. Frontiers in Environmental Science. https://doi.org/10.3389/fenvs.2020.572424
- Feride Öykü Sefiloğlu et al. (2024). *Quantitative analysis of microplastics in Nile tilapia from a recirculating aquaculture system using pyrolysis–gas chromatography–mass spectrometry*. Environmental Sciences Europe. https://doi.org/10.1186/s12302-024-00987-6
- Flavia Capuozzo et al. (2025). *Evaluation of Digestion Methods in Microplastic Recovery from Mussels (Mytilus galloprovincialis) for a Standardised Microplastic Isolation Protocol*. Foods. https://doi.org/10.3390/foods14223853
- Florence Nono Almeida et al. (2025). *Py-GC-MS/MS quantification of microplastics in vertebrate tissues: Addressing false positives of polyethylene*. Journal of Hazardous Materials. https://doi.org/10.1016/j.jhazmat.2025.140658
- Francisco Alvirde-Díaz et al. (2026). *Determination of the Best Digestion and Extraction Methods for the Quantification of Microplastics in Landfill Leachate*. Microplastics. https://doi.org/10.3390/microplastics5030134
- Gabriel Erni-Cassola et al. (2017). *Lost, but Found with Nile Red: A Novel Method for Detecting and Quantifying Small Microplastics (1 mm to 20 μm) in Environmental Samples*. Environmental Science & Technology. https://doi.org/10.1021/acs.est.7b04512
- Gabriela Fernandez et al. (2026). *Advancing marine microplastic monitoring through deep learning-based image segmentation*. One Health Advances. https://doi.org/10.1186/s44280-026-00120-7
- Giuseppe Egidio De Benedetto et al. (2024). *Microplastics determination and quantification in two benthic filter feeders Sabella spallanzanii, Polychaeta and Paraleucilla magna, Porifera*. Heliyon. https://doi.org/10.1016/j.heliyon.2024.e31796
- Giusto Lo Bue et al. (2026). *Microplastic quantification in Sabellaria reefs: a validated protocol for extraction from biogenic agglutinated matrices*. Environmental Science and Pollution Research. https://doi.org/10.1007/s11356-026-37573-7
- Grzegorz Sychowski et al. (2025). *Microplastics Exposure Impact on Lung Cancer—Literature Review*. Cancers. https://doi.org/10.3390/cancers17223616
- Gustavo R. Bitencourt et al. (2020). *Determination of microplastic content in seafood: An integrated approach combined with the determination of elemental contaminants*. The Science of The Total Environment. https://doi.org/10.1016/j.scitotenv.2020.142301
- Günay Yıldız Töre & Şirin İLHAN YILMAZ (2022). *Detection and Mitigation of Microplastics (MPs) in Environment and Applications on Legal Regulations–A Review*. Journal of Balkan Science and Technology. https://doi.org/10.55848/jbst.2022.6
- H. Yahya et al. (2023). *Occurrence and pathways of microplastics, quantification protocol and adverse effects of microplastics towards freshwater and seawater biota*. Food Research. https://doi.org/10.26656/fr.2017.7(5).133
- Haesung Lee et al. (2023). *Pretreatment methods for monitoring microplastics in soil and freshwater sediment samples: A comprehensive review*. The Science of The Total Environment. https://doi.org/10.1016/j.scitotenv.2023.161718
- Haesung Lee et al. (2024). *Establishing freshwater sediment sample pretreatment methods for monitoring microplastics and its challenges*. Chemical Engineering Journal. https://doi.org/10.1016/j.cej.2024.152059
- Hai Ngoc Dinh et al. (2023). *Comparision protocols for extraction of microplastics in water samples*. Vietnam Journal of Marine Science and Technology. https://doi.org/10.15625/1859-3097/17430
- He Zhu, Jing Luo & Sailing He (2026). *Label-free 4D confocal Raman imaging enabling in vivo multiplex identification of microplastics in zebrafish*. Journal of Hazardous Materials. https://doi.org/10.1016/j.jhazmat.2026.143319
- Heba Elbasiouny & Fathy Elbehiry (2023). *Addressing the Microplastic Dilemma in Soil and Sediment with Focus on Biochar-Based Remediation Techniques: Review*. Soil Systems. https://doi.org/10.3390/soilsystems7040110
- Heejun Kang et al. (2020). *Modification of a Nile Red Staining Method for Microplastics Analysis: A Nile Red Plate Method*. Water. https://doi.org/10.3390/w12113251
- Helge Hove, Thomas Næsheim & Tanja Kögel (2023). *Quick and efficient microplastic isolation from fatty fish tissues by surfactant-enhanced alkaline digestion*. Marine Pollution Bulletin. https://doi.org/10.1016/j.marpolbul.2023.115726
- Hewawasam Udumullage Erangi Imasha et al. (2025). *Integrated Impacts of Microplastics Contaminated Compost on Soil–Plant Systems and Bioaccumulation in Lettuce, Tomato, and Radish*. Water Air & Soil Pollution. https://doi.org/10.1007/s11270-025-08684-7
- Hiraku Tanoiri et al. (2023). *The pretreatment method in marine organisms and sediment for microplastics analysis by FTIR using “Cylindrical microplastics fractionator”*. MethodsX. https://doi.org/10.1016/j.mex.2023.102396
- Hiraku Tanoiri et al. (2026). *Replacing human judgment in fluorescence-based microplastic detection with machine learning: toward harmonization with spectroscopic methods*. Journal of Hazardous Materials Advances. https://doi.org/10.1016/j.hazadv.2026.101050
- Homin Kye et al. (2023). *Microplastics in water systems: A review of their impacts on the environment and their potential hazards*. Heliyon. https://doi.org/10.1016/j.heliyon.2023.e14359
- Hrithik Nath et al. (2026). *Organic matter removal in microplastic extraction from environmental samples: Challenges and methodological solutions*. Journal of Hazardous Materials. https://doi.org/10.1016/j.jhazmat.2026.142998
- Hsuan-Cheng Lu et al. (2021). *A systematic review of freshwater microplastics in water and sediments: Recommendations for harmonisation to enhance future study comparisons*. The Science of The Total Environment. https://doi.org/10.1016/j.scitotenv.2021.146693
- Imke Podbielski, Thea Hamm & Mark Lenz (2024). *Customized digestion protocols for copepods, euphausiids, chaetognaths and fish larvae facilitate the isolation of ingested microplastics*. Scientific Reports. https://doi.org/10.1038/s41598-024-70366-x
- Irene Dini, Andrea Mancusi & Serenella Seccia (2025). *From Harm to Hope: Tackling Microplastics’ Perils with Recycling Innovation*. Molecules. https://doi.org/10.3390/molecules30122535
- Irfan Hassan et al. (2025). *Tracking microplastics at the source: a comparative study of fluorescent and FTIR microscopy at a drinking water intake in the Perak River, Malaysia*. Environmental Monitoring and Assessment. https://doi.org/10.1007/s10661-025-14127-x
- Irène Godéré et al. (2026). *Exploring Nile Red and machine learning for microplastics detection in Tridacna maxima*. PLoS ONE. https://doi.org/10.1371/journal.pone.0357014
- Isidora Catalán-Muñoz et al. (2026). *Plastic microfibers in water, macroinvertebrates and Chilean pencil catfish across the Mapocho River basin (Chile): Occurrence, characteristics, exposure and trophic transfer trends*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2026.128034
- Ivanilson da Silva de Aquino et al. (2026). *Development of a Methodology for Analyzing Microplastics in Sewage Treatment Plants*. Archives of Environmental Contamination and Toxicology. https://doi.org/10.1007/s00244-026-01208-2
- Jaden Tolbert, Islam, Md Saiful & Pingshan Wang (2026). *Radio Frequency Detection and Classification of Microplastics in Water*. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2609.20507
- James V. Tarte et al. (2025). *Large Scale Microplastic Fibre Analysis in Wastewater: A Comprehensive Review and Recommendations*. Current Pollution Reports. https://doi.org/10.1007/s40726-025-00378-2
- Jan Halfar et al. (2021). *Disparities in Methods Used to Determine Microplastics in the Aquatic Environment: A Review of Legislation, Sampling Process and Instrumental Analysis*. International Journal of Environmental Research and Public Health. https://doi.org/10.3390/ijerph18147608
- Jan Kamp et al. (2023). *Quantification of Poly(vinyl chloride) Microplastics via Pressurized Liquid Extraction and Combustion Ion Chromatography*. Environmental Science & Technology. https://doi.org/10.1021/acs.est.2c06555
- Javier Lorenzo-Navarro et al. (2020). *SMACC: A System for Microplastics Automatic Counting and Classification*. IEEE Access. https://doi.org/10.1109/access.2020.2970498
- Jayasree Kumar et al. (2025). *Trapping tiny pollutants: SERS-driven strategies for microplastics and nanoplastics detection*. iScience. https://doi.org/10.1016/j.isci.2025.113888
- Jean Yves Uwamungu et al. (2022). *Microplastic contamination in soil agro-ecosystems: A review*. Environmental Advances. https://doi.org/10.1016/j.envadv.2022.100273
- Jeanette Lykkemark et al. (2024). *Py–GC–MS analysis for microplastics: Unlocking matrix challenges and sample recovery when analyzing wastewater for polypropylene and polystyrene*. Water Research. https://doi.org/10.1016/j.watres.2024.122055
- Jenevieve Hara, João P.G.L. Frias & Róisín Nash (2020). *Quantification of microplastic ingestion by the decapod crustacean Nephrops norvegicus from Irish waters*. Marine Pollution Bulletin. https://doi.org/10.1016/j.marpolbul.2020.110905
- Jennifer Paola Pascali et al. (2026). *A Robust Multistep Digestion Method for Microplastics Detection in Human Tissue by MicroRaman Analysis*. Drug Testing and Analysis. https://doi.org/10.1002/dta.70024
- Jhao-You Lin, Bishwatma Biswas & Yi-Pin Lin (2026). *A fluorescence-based protocol for quantifying microplastics in soil: Protocol optimization and field investigation*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2026.129104
- Jianqiao Song, Kaushik Venkiteshwaran & Shenghua Wu (2026). *Artificial intelligence in microplastic and nanoplastic research: State-of-Art review from automated detection to environmental fate modeling*. Journal of Hazardous Materials Plastics. https://doi.org/10.1016/j.hazmp.2026.100083
- Jiehan Duan et al. (2019). *Development of a digestion method for determining microplastic pollution in vegetal-rich clayey mangrove sediments*. The Science of The Total Environment. https://doi.org/10.1016/j.scitotenv.2019.136030
- Joan Y. Q. Li, Leo Nankervis & Amanda  Louise Dawson (2022). *Digesting the Indigestible: Microplastic Extraction From Prawn Digestive Tracts*. Frontiers in Environmental Chemistry. https://doi.org/10.3389/fenvc.2022.903314
- Joana C. Prata et al. (2018). *Methods for sampling and detection of microplastics in water and sediment: A critical review*. TrAC Trends in Analytical Chemistry. https://doi.org/10.1016/j.trac.2018.10.029
- Julene Aramendia et al. (2024). *Evidence of internalized microplastics in mussel tissues detected by volumetric Raman imaging*. The Science of The Total Environment. https://doi.org/10.1016/j.scitotenv.2024.169960
- Julia Süssmann et al. (2021). *Evaluation and optimisation of sample preparation protocols suitable for the analysis of plastic particles present in seafood*. Food Control. https://doi.org/10.1016/j.foodcont.2021.107969
- Jun‐Ray Macairan et al. (2023). *Tissue Clearing To Localize Microplastics via Three-Dimensional Imaging of Whole Organisms*. Environmental Science & Technology. https://doi.org/10.1021/acs.est.2c07209
- Kalman Kovács et al. (2026). *Microplastics in human ovarian follicular fluid: a pilot Raman spectroscopy study*. Frontiers in Endocrinology. https://doi.org/10.3389/fendo.2026.1856825
- Katherine K. Porterfield et al. (2023). *Microplastics in composts, digestates, and food wastes: A review*. Journal of Environmental Quality. https://doi.org/10.1002/jeq2.20450
- Kexin Li et al. (2026). *Medical Microplastics: Research Progress on Exposure Pathways, Toxic Effects, and Detection Methods*. Microplastics. https://doi.org/10.3390/microplastics5020061
- Kleopatra Miserli & Ioannis Konstantinou (2025). *Determination of microplastics in municipal wastewater treatment plant effluents and sludge using micro-Raman spectroscopy*. Water Emerging Contaminants & Nanoplastics. https://doi.org/10.20517/wecn.2024.72
- Konstantinos Tsamoutsoglou, Vasileia Gkika & Li Xu (2026). *AI-assisted microplastics detection, removal and management: Advances and challenges in wastewater*. Artificial Intelligence & Environment. https://doi.org/10.66178/aie-0026-0012
- Kornelia Kadac-Czapska et al. (2024). *Methodology Approach for Microplastics Isolation from Samples Containing Sucrose*. Molecules. https://doi.org/10.3390/molecules29173996
- Kristina Duswald et al. (2024). *Detection of Unlabeled Micro- and Nanoplastics in Unstained Tissue with Optical Photothermal Infrared Spectroscopy*. bioRxiv (Cold Spring Harbor Laboratory). https://doi.org/10.1101/2024.11.11.622943
- Leila Shafea et al. (2026). *Microplastics recovery from soil by density separation: application at pristine and UV-aged particles differing in surface properties*. Microplastics and Nanoplastics. https://doi.org/10.1186/s43591-026-00194-5
- Lency Rosana et al. (2026). *Distribution, abundance, and characteristics of microplastics in surficial beach sediments of Miagao, Iloilo*. SciEnggJ. https://doi.org/10.54645/202619suplnf-24
- Lin Wang & Hemin Zhang (2026). *A Review of Environmental Microplastic Detection Methods*. Journal of Computer Science and Artificial Intelligence. https://doi.org/10.54097/x0sx7j78
- Linn Merethe Brekke Olsen et al. (2020). *Facilitating microplastic quantification through the introduction of a cellulose dissolution step prior to oxidation: Proof-of-concept and demonstration using diverse samples from the Inner Oslofjord, Norway*. Marine Environmental Research. https://doi.org/10.1016/j.marenvres.2020.105080
- Liyuan Gong (2025). *MICROPLASTICS DETECTION, CHARACTERIZATION, AND CYTOTOXICITY EVALUATION VIA LAB-ON-CHIP*. . https://doi.org/10.23860/diss-4453
- Liyuan Gong et al. (2023). *A microfluidic approach for label-free identification of small-sized microplastics in seawater*. Scientific Reports. https://doi.org/10.1038/s41598-023-37900-9
- Ludovic Dulac et al. (2026). *Overcoming matrix interference: a selective fluorescence method for the detection and quantification of PHA, PLA, and PBAT microplastics in soil*. Polymer Testing. https://doi.org/10.1016/j.polymertesting.2026.109300
- Ludovic Hermabessière & Chelsea M. Rochman (2021). *Microwave-Assisted Extraction for Quantification of Microplastics Using Pyrolysis–Gas Chromatography/Mass Spectrometry*. Environmental Toxicology and Chemistry. https://doi.org/10.1002/etc.5179
- Lukas Brunnbauer et al. (2026). *A Method for the Detection of Tire Wear Microplastics in Zebrafish Guts by Laterally Resolved LA-ICP-MS-Based Elemental Fingerprinting and Chemometrics*. Analytical Chemistry. https://doi.org/10.1021/acs.analchem.5c07035
- Lulu Lv et al. (2019). *A simple method for detecting and quantifying microplastics utilizing fluorescent dyes - Safranine T, fluorescein isophosphate, Nile red based on thermal expansion and contraction property*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2019.113283
- Mahima Doshi, Vasantkumar Rabari & Jigneshkumar Narmadashankar Trivedi (2025). *A systematic review on microplastic contamination in marine sediment and water of Asia: Concentration, characterization, and polymeric risk assessment*. Water Environment Research. https://doi.org/10.1002/wer.70042
- Mahmoud Sami, Sara El-Kerkary & Marina S. Fawzy (2026). *Developing and evaluating automated deep learning and human-in-the-loop vision–language systems for microplastic characterization*. Scientific Reports. https://doi.org/10.1038/s41598-026-67536-4
- Manca Kovač Viršek et al. (2016). *Protocol for Microplastics Sampling on the Sea Surface and Sample Analysis*. Journal of Visualized Experiments. https://doi.org/10.3791/55161-v
- Mara Putzu et al. (2025). *Accuracy assessment of a micro-Raman spectroscopy method for small microplastic particles in infant milk formula*. Talanta Open. https://doi.org/10.1016/j.talo.2025.100586
- Marco Perez et al. (2024). *Development of a new methodology for the determination of PET microplastics in sediment, based on microwave-assisted acid digestion*. PLoS ONE. https://doi.org/10.1371/journal.pone.0314520
- Marlo Baker et al. (2025). *A Survey of Microplastics in Puerto Rican Sands using NaCl solution separation methods and Nile red optical staining*. Abstracts with programs - Geological Society of America. https://doi.org/10.1130/abs/2025am-8026
- Marlo Baker et al. (2026). *A survey of microplastics in Puerto Rican sands using sodium chloride and zinc chloride solution separation methods and Nile red optical staining*. Abstracts with programs - Geological Society of America. https://doi.org/10.1130/abs/2026se-12567
- Marshall W. Ritchie et al. (2022). *Characterizing microplastic ingestion, transformation, and excretion in insects using fluorescent plastics*. bioRxiv (Cold Spring Harbor Laboratory). https://doi.org/10.1101/2022.11.11.516210
- Martin Šteković et al. (2026). *Development of a Microwave-Assisted Digestion Procedure for Microplastics Extraction from Different Food Matrices with Subsequent Analysis Using Raman Microspectroscopy*. Microplastics. https://doi.org/10.3390/microplastics5010045
- Mary Carolin Kurisingal Cleetus et al. (2026). *Challenges in the analysis of microplastics and nanoplastics extracted from marine biological tissues: a review*. Environmental Chemistry Letters. https://doi.org/10.1007/s10311-026-01902-6
- Masashi Tsuchiya et al. (2025). *Rapid detection and quantification of Nile Red-stained microplastic particles in sediment samples*. PeerJ. https://doi.org/10.7717/peerj.19196
- Matthias Tamminga et al. (2017). *Nile Red Staining as a Subsidiary Method for Microplastic Quantifica-tion: A Comparison of Three Solvents and Factors Influencing Application Reliability*. SDRP Journal of Earth Sciences & Environmental Studies. https://doi.org/10.25177/jeses.2.2.1
- Max Beaurepaire (2024). *Contributing to the assessment of the impact of urban activities on microplastic transport through air and runoff infiltration*. . https://doi.org/10.70675/299008b3z75fcz40cbzb545z3c9b6293602f
- Maximiliano Campos-López et al. (2024). *Advancing Microplastic Detection Technology through Digital Image Processing, Fractal Analysis, and Polynomial Approximation Methods*. Microscopy and Microanalysis. https://doi.org/10.1093/mam/ozae044.195
- Md Abdul Baset Sarker et al. (2024). *Real-Time Detection of Microplastics Using an AI Camera*. Sensors. https://doi.org/10.3390/s24134394
- Md Ruhul Amin et al. (2026). *Spatio-temporal dynamics and risk assessment of microplastics in water, sediment and fish from a critical wetland ecosystem: Arial Beel, Bangladesh*. Journal of Hazardous Materials Advances. https://doi.org/10.1016/j.hazadv.2026.101373
- Meiqing Jin et al. (2022). *Current development and future challenges in microplastic detection techniques: A bibliometrics-based analysis and review*. Science Progress. https://doi.org/10.1177/00368504221132151
- Meiqing Jin et al. (2025). *Microplastic contamination in sediments: Analytical techniques and case-based evaluations*. Talanta. https://doi.org/10.1016/j.talanta.2025.128267
- Melina S. Stathopoulou et al. (2026). *Methods for detecting microplastics in human tissues, with a focus on vascular and intravascular specimens*. Archives of Medical Science - Atherosclerotic Diseases. https://doi.org/10.5114/amsad/226847
- Mengjie Wu et al. (2020). *Microplastics in waters and soils: Occurrence, analytical methods and ecotoxicological effects*. Ecotoxicology and Environmental Safety. https://doi.org/10.1016/j.ecoenv.2020.110910
- Michael Schwarzer et al. (2026). *Effects of petroleum-based and biodegradable bio-based microplastics versus natural control particles on Paramecium caudatum, combined with in situ Raman spectroscopic detection*. Journal of Hazardous Materials. https://doi.org/10.1016/j.jhazmat.2026.141575
- Michael Sturm et al. (2023). *Development of an Inexpensive and Comparable Microplastic Detection Method Using Fluorescent Staining with Novel Nile Red Derivatives*. Analytica—A Journal of Analytical Chemistry and Chemical Analysis. https://doi.org/10.3390/analytica4010004
- Michael Sturm et al. (2023). *Fast Forward: Optimized Sample Preparation and Fluorescent Staining for Microplastic Detection*. Microplastics. https://doi.org/10.3390/microplastics2040026
- Michael Sturm et al. (2026). *Improving Sampling Strategies for Microplastic Detection in Aquatic Environments: Precision, Recovery, and Sample Size Requirements*. Microplastics. https://doi.org/10.3390/microplastics5020075
- Mikaël Kedzierski et al. (2016). *Microplastics elutriation from sandy sediments: A granulometric approach*. Marine Pollution Bulletin. https://doi.org/10.1016/j.marpolbul.2016.03.041
- Mingtan Dong et al. (2021). *Automated analysis of microplastics based on vibrational spectroscopy: Are we measuring the same metrics?*. . https://doi.org/10.33774/chemrxiv-2021-hqr34-v2
- Mirella Peña-Icart et al. (2011). *Comparative study of digestion methods EPA 3050B (HNO3–H2O2–HCl) and ISO 11466.3 (aqua regia) for Cu, Ni and Pb contamination assessment in marine sediments*. Marine Environmental Research. https://doi.org/10.1016/j.marenvres.2011.05.005
- Mirka Lares et al. (2018). *Occurrence, identification and removal of microplastic particles and fibers in conventional activated sludge process and advanced MBR technology*. Water Research. https://doi.org/10.1016/j.watres.2018.01.049
- Mohammed Al-Azzawi et al. (2020). *Validation of Sample Preparation Methods for Microplastic Analysis in Wastewater Matrices—Reproducibility and Standardization*. Water. https://doi.org/10.3390/w12092445
- Monica G. da Silva et al. (2026). *Hepatic Mitochondrial Dysfunction and Gut Dysbiosis Induced by Polyethylene Microplastics in FVB/n Mice: A Comparative Study of Fluorescent and Non-Fluorescent Particles*. Toxics. https://doi.org/10.3390/toxics14050386
- Montaha Behbehani et al. (2025). *Sample Collection, Preparation, and Identification of Microplastics in Soil: Issues and Recommendations*. Sustainability Sciences in Asia and Africa. https://doi.org/10.1007/978-981-95-2740-3_3
- Muhammad Bilal Shakoor et al. (2026). *Engineered biochar as a powerful tool for microplastics removal from wastewater: a critical review*. Environmental Geochemistry and Health. https://doi.org/10.1007/s10653-026-03414-2
- Musthofa Galih Pradana (2026). *YOLOv8-Based Microplastic Detection and Quantification in River Water Microscopic Images*. Journal of Applied Data Sciences. https://doi.org/10.47738/jads.v7i2.1262
- Nam Ngoc Phuong et al. (2021). *Highlights from a review of microplastics in marine sediments*. The Science of The Total Environment. https://doi.org/10.1016/j.scitotenv.2021.146225
- Namrata Kaile et al. (2020). *Preliminary Results From Detection of Microplastics in Liquid Samples Using Flow Cytometry*. Frontiers in Marine Science. https://doi.org/10.3389/fmars.2020.552688
- Nathan Manimesia Kayemba et al. (2026). *Assessment of Microplastic Contamination in Surface Waters and Sediments of Pool Malebo, Congo River, Democratic Republic of the Congo*. Natural Built Social Environment Health. https://doi.org/10.63095/nbseh.26.551063
- Neeraj Kumar et al. (2023). *Instrumental Detection Of Microplastics In Fish And Their Harmful Effects On Human Health*. . https://doi.org/10.53555/sfs.v10i1.3386
- Nelle Meyers et al. (2022). *Microplastic detection and identification by Nile red staining: Towards a semi-automated, cost- and time-effective technique*. The Science of The Total Environment. https://doi.org/10.1016/j.scitotenv.2022.153441
- Nelle Meyers et al. (2024). *Towards reliable data: Validation of a machine learning-based approach for microplastics analysis in marine organisms using Nile red staining*. Marine Pollution Bulletin. https://doi.org/10.1016/j.marpolbul.2024.116804
- NGIE-HAO ERIC LAU et al. (2025). *Microplastic Quantification in Meretrix lyrata through Rapid Screening Method using Nile Red*. Borneo Journal of Resource Science and Technology. https://doi.org/10.33736/bjrst.8508.2025
- Nick Krekelbergh et al. (2025). *Comparison of Raman and fluorescence microscopy for identification of small (< 2 μm) microplastics in soil*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2025.126204
- Niu Imeleta Faauma et al. (2026). *Beyond Removal: A Critical Review of Microplastic Mass Flux, In-Plant Transformation, and Elimination in WWTPs*. Molecules. https://doi.org/10.3390/molecules31050798
- Noah A. D’Ascanio, Husein Almuhtaram & Robert C. Andrews (2025). *Selection of an appropriate fluorescent reference material to assess microplastic recovery in natural waters*. Microplastics and Nanoplastics. https://doi.org/10.1186/s43591-025-00125-w
- Nur Sakinah Roslan et al. (2024). *Detection of microplastics in human tissues and organs: A scoping review*. Journal of Global Health. https://doi.org/10.7189/jogh.14.04179
- Oliver Jacob et al. (2024). *Towards a reference material for microplastics’ number concentration—case study of PET in water using Raman microspectroscopy*. Analytical and Bioanalytical Chemistry. https://doi.org/10.1007/s00216-024-05251-7
- Olivia Gérigny et al. (2024). *Comparative analysis of microplastics detection methods applied to marine sediments: A case study in the Bay of Marseille*. Marine Pollution Bulletin. https://doi.org/10.1016/j.marpolbul.2024.116787
- Orlagh Gormley et al. (2026). *Micro- and nanoplastics in food and water from farm-to-fork: exposure pathways, human biomonitoring and AI-enhanced detection – a systematic review*. npj Science of Food. https://doi.org/10.1038/s41538-026-01054-5
- Padinchare Veettil Gayathri et al. (2023). *Advanced oxidation processes for the degradation of microplastics from the environment: A review*. Water and Environment Journal. https://doi.org/10.1111/wej.12883
- Paniteja Madala, Amey Waikar & Hemraj Parate (2025). *Detection to Remediation: Strategies for Managing Microplastic Pollution in Freshwater Systems*. International Journal of Computational and Experimental Science and Engineering. https://doi.org/10.22399/ijcesen.3452
- Patrizia Marie Pfohl et al. (2021). *Microplastic extraction protocols can impact the polymer structure*. Microplastics and Nanoplastics. https://doi.org/10.1186/s43591-021-00009-9
- Paula Przygoda-Kuś et al. (2025). *Current Approaches to Microplastics Detection and Plastic Biodegradation*. Molecules. https://doi.org/10.3390/molecules30112462
- Pengfei Wu et al. (2023). *Machine Learning-Assisted Insights into Sources and Fate of Microplastics in Wastewater Treatment Plants*. ACS ES&T Water. https://doi.org/10.1021/acsestwater.3c00386
- Peter Vermeiren, Cynthia C. Muñoz & Kou Ikejima (2020). *Microplastic identification and quantification from organic rich sediments: A validated laboratory protocol*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2020.114298
- Petros Chatzitoulousis et al. (2026). *A pipeline for meso- and microplastic identification in aquatic systems using machine learning and hyperspectral imaging*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2026.127995
- Pierluigi Strafella et al. (2024). *Detection of microplastics in fish using computed tomography and deep learning*. Heliyon. https://doi.org/10.1016/j.heliyon.2024.e39875
- Poramee Khongmeunrak et al. (2025). *Standardized trimodal histopathological examination for microplastic detection and tissue-level assessment in green mussels (Perna viridis) cultivated near an industrial estate in Rayong, Thailand*. Veterinary World. https://doi.org/10.14202/vetworld.2025.4212-4235
- Qiannan Duan et al. (2026). *Fluorescence Machine Vision-Based Rapid Quantitative Characterization of Microplastics*. Analytical Chemistry. https://doi.org/10.1021/acs.analchem.6c00407
- Rajeshwori Malla-Pradhan et al. (2023). *Microplastic in Freshwater Environment: A Review on Techniques and Abundance for Microplastic Detection in Lake Water*. Trends in Sciences. https://doi.org/10.48048/tis.2023.5202
- RAJU KRISHNA CHALANNAVAR et al. (2025). *Microplastics: Detection methods-An update*. World Journal of Advanced Research and Reviews. https://doi.org/10.30574/wjarr.2025.26.2.1715
- Rea Grace Galit et al. (2026). *Environmental distribution and polymer characterization of microplastics in the Miagao Estuary, Iloilo, Philippines*. The Palawan Scientist. https://doi.org/10.69721/tps.j.2026.18.2.05
- Riley Moran (2026). *RAPID MICROPLASTIC PROCESSING FOR HIGH SPATIOTEMPORAL RESOLUTION: APPLICATIONS OF NOVEL FLUORESENCE EQUIMENT AND METHODS TO HISTORICAL MICROPLASTIC ACCUMULATION IN JORDAN LAKE*. Carolina Digital Repository (University of North Carolina at Chapel Hill). https://doi.org/10.17615/ff8k-ez06
- Risky Ayu Kristanti et al. (2022). *Characteristics of Microplastic in Commercial Aquatic Organisms*. Tropical Aquatic and Soil Pollution. https://doi.org/10.53623/tasp.v2i2.134
- Roberto Nisticò (2020). *Polyethylene terephthalate (PET) in the packaging industry*. Polymer Testing. https://doi.org/10.1016/j.polymertesting.2020.106707
- Rongrong Song et al. (2026). *Microwave-assisted digestion for rapid extraction of microplastics from food matrices*. Journal of Food Composition and Analysis. https://doi.org/10.1016/j.jfca.2026.109485
- Rubén Rodríguez-Alegre et al. (2025). *Comparative Assessment of Protocols for Microplastic Quantification in Wastewater*. Microplastics. https://doi.org/10.3390/microplastics4030049
- S. Facchetti et al. (2020). *Detection of Metal-Doped Fluorescent PVC Microplastics in Freshwater Mussels*. Nanomaterials. https://doi.org/10.3390/nano10122363
- S. Lynch et al. (2025). *An integrated, tiered microplastic workflow, supporting rapid broadscale detection options*. MethodsX. https://doi.org/10.1016/j.mex.2025.103536
- Sabela Fernandez-Sanchez et al. (2026). *Detection of Nanoplastics in Marine Environments: Current Methods and Future Perspectives*. Microplastics. https://doi.org/10.3390/microplastics5020121
- Sadia A. Younis et al. (2026). *The hidden hazard in every sip: tackling microplastics in drinking water through detection and mitigation*. Food Chemistry X. https://doi.org/10.1016/j.fochx.2026.104415
- Saidu Garba et al. (2026). *Occurrence and Characterisation of Microplastics in Fish Tissues and Water from Yusuf Bage River, Unguwar Rimi, Kaduna State, Nigeria*. FUDMA Journal of Sciences. https://doi.org/10.33003/fjs-2026-1020-6266
- Saif Uddin et al. (2025). *Sources of Microplastics in the Soil and Use of Laser Direct Infrared Imaging for Microplastic Detection*. Sustainability Sciences in Asia and Africa. https://doi.org/10.1007/978-981-95-2740-3_6
- Samantha Phan & Christine Keiko Luscombe (2023). *Recent trends in marine microplastic modeling and machine learning tools: Potential for long-term microplastic monitoring*. Journal of Applied Physics. https://doi.org/10.1063/5.0126358
- Samantha Phan et al. (2023). *Exploiting weak supervision to facilitate segmentation, classification, and analysis of microplastics (<100 μm) using Raman microspectroscopy images*. The Science of The Total Environment. https://doi.org/10.1016/j.scitotenv.2023.163786
- Sarah Alotaibi, Gaddi B. Eshun & Omowunmi A. Sadik (2026). *Tracking Microplastics and Their Associated Chemical Additives in Plant Tissues: A Pyrolysis GC-MS Approach to Identification, Quantification, and Translocation Mechanism*. ACS Omega. https://doi.org/10.1021/acsomega.5c08234
- Sarah‐Jeanne Royer et al. (2024). *Computer vision segmentation model—deep learning for categorizing microplastic debris*. Frontiers in Environmental Science. https://doi.org/10.3389/fenvs.2024.1386292
- Sebastiano La Maestra et al. (2025). *Microplastics as benzo-a-pyrene carriers: genotoxicity assessment simulating human gastric digestion*. Archives of Toxicology. https://doi.org/10.1007/s00204-025-04046-8
- Sedat Gündoğdu et al. (2025). *Microplastics in soil: a comprehensive review of analytical techniques*. Frontiers in Soil Science. https://doi.org/10.3389/fsoil.2025.1614075
- Sergio Martínez-Navas et al. (2026). *Dual-Channel fluorescent sensor for in-situ microplastic detection in marine environments*. Results in Engineering. https://doi.org/10.1016/j.rineng.2026.112175
- Sheen Mclean Cabaneros et al. (2025). *Automatic pre-screening of outdoor airborne microplastics in micrographs using deep learning*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2025.125993
- Shiwani Shiwani et al. (2026). *Characterizing the Influence of Filter Substrates on Raman Measurements of Microplastics*. Analytical Chemistry. https://doi.org/10.1021/acs.analchem.6c02679
- Shramana Koner & Ramasubbu Seenivasan (2025). *A review on decoding the cellular and ecological risks of polypropylene microplastics*. Environmental Pollutants and Bioavailability. https://doi.org/10.1080/26395940.2025.2583466
- Silvia Galafassi et al. (2021). *Do Freshwater Fish Eat Microplastics? A Review with A Focus on Effects on Fish Health and Predictive Traits of MPs Ingestion*. Water. https://doi.org/10.3390/w13162214
- Sneha Siwach et al. (2026). *A novel non-invasive imaging technique for mapping microplastic accumulation and distribution in different organ systems of Indian major carp Labeo rohita (Ham. 1822)*. Current Research in Toxicology. https://doi.org/10.1016/j.crtox.2026.100294
- Sonya R. Moses et al. (2023). *Comparison of two rapid automated analysis tools for large FTIR microplastic datasets*. Analytical and Bioanalytical Chemistry. https://doi.org/10.1007/s00216-023-04630-w
- Soyeon Jeon et al. (2026). *A simple and reliable method for the qualitative and quantitative analysis of nano- and microplastics in organs for controlled preclinical studies*. Particle and Fibre Toxicology. https://doi.org/10.1186/s12989-026-00677-6
- Soyoun Kim, Yejin Hyeon & Chanhyuk Park (2023). *Microplastics’ Shape and Morphology Analysis in the Presence of Natural Organic Matter Using Flow Imaging Microscopy*. Molecules. https://doi.org/10.3390/molecules28196913
- Sree Sesha Sravanika Bezawada et al. (2026). *A Review on Microplastics: The Emerging Threat to Food Safety*. Nature Environment and Pollution Technology. https://doi.org/10.46488/nept.2026.v25i02.b4384
- Srumika Konde et al. (2023). *Nile Red staining for the detection of microplastics: a comprehensive study on the emission spectra*. Research Square. https://doi.org/10.21203/rs.3.rs-2579237/v1
- Stefania Mariano et al. (2021). *Micro and Nanoplastics Identification: Classic Methods and Innovative Detection Techniques*. Frontiers in Toxicology. https://doi.org/10.3389/ftox.2021.636640
- Steve Utecht, Stefan Krause & Tobias Schuetz (2025). *Particle size-dependent quantitative and qualitative differences of common microplastic detection procedures: Nile Red-assisted fluorescence microscopy and confocal micro-Raman spectroscopy*. Environmental Monitoring and Assessment. https://doi.org/10.1007/s10661-025-14317-7
- Suparnamaaya Prasad, Andrew Arnold Bennett & Michael S. Triantafyllou (2024). *Characterization of Nile Red-Stained Microplastics through Fluorescence Spectroscopy*. Journal of Marine Science and Engineering. https://doi.org/10.20944/preprints202407.2045.v1
- Suparnamaaya Prasad, Andrew Arnold Bennett & Michael S. Triantafyllou (2024). *Characterization of Nile Red-Stained Microplastics through Fluorescence Spectroscopy*. Journal of Marine Science and Engineering. https://doi.org/10.3390/jmse12081403
- Susanne Richter et al. (2022). *A reference methodology for microplastic particle size distribution analysis: Sampling, filtration, and detection by optical microscopy and image processing*. Applied Research. https://doi.org/10.1002/appl.202200055
- Tabea Scheiterlein & Peter Fiener (2026). *Microplastics detection in agricultural soil combining 3D Laser Scanning Confocal Microscopy with machine learning*. . https://doi.org/10.5194/egusphere-egu26-5293
- Tabea Scheiterlein & Peter Fiener (2026). *Microplastics detection in agricultural soil combining 3D Laser Scanning Confocal Microscopy with machine learning*. Microplastics and Nanoplastics. https://doi.org/10.1186/s43591-026-00202-8
- Tabea Scheiterlein & Peter Fiener (2026). *Microplastics detection in agricultural soil combining 3D Laser Scanning Confocal Microscopy with machine learning*. Microplastics and Nanoplastics. https://doi.org/10.21203/rs.3.rs-8373800/v1
- Taher M. Ghazal et al. (2025). *Enhanced eDNA Recovery from Microplastic-Polluted Freshwater Systems Using Surfactant-Assisted Bead-Beating with Enzymatic Digestion*. Natural and Engineering Sciences. https://doi.org/10.28978/nesciences.1811139
- Tanja Bogdanović et al. (2025). *Microplastic Polymer Mass Fractions in Marine Bivalves: From Isolation to Hazard Risk*. Journal of Xenobiotics. https://doi.org/10.3390/jox15060186
- Thomas Maes et al. (2017). *A rapid-screening approach to detect and quantify microplastics based on fluorescent tagging with Nile Red*. Scientific Reports. https://doi.org/10.1038/srep44501
- Thuhin Kumar Dey, Md. Elias Uddin & Mamun Jamal (2021). *Detection and removal of microplastics in wastewater: evolution and impact*. Environmental Science and Pollution Research. https://doi.org/10.1007/s11356-021-12943-5
- Thunchanok Thammasanya et al. (2024). *A new approach to classifying polymer type of microplastics based on Faster-RCNN-FPN and spectroscopic imagery under ultraviolet light*. Scientific Reports. https://doi.org/10.1038/s41598-024-53251-5
- Tianci Li et al. (2026). *Sex-specific feeding ecology and microplastic ingestion in smooth hammerhead shark (Sphyrna zygaena) from the eastern tropical Pacific Ocean: insights from stable isotope and fatty acid signatures*. Marine Environmental Research. https://doi.org/10.1016/j.marenvres.2026.108311
- Trishan Naidoo, K. Goordiyal & David Glassom (2017). *Are Nitric Acid (HNO3) Digestions Efficient in Isolating Microplastics from Juvenile Fish?*. Water Air & Soil Pollution. https://doi.org/10.1007/s11270-017-3654-4
- Tuncay VURAL et al. (2025). *Protocol for extraction and analysis of microplastics in freshwater, sediment, and fish samples*. STAR Protocols. https://doi.org/10.1016/j.xpro.2025.104057
- Un‐Jung Kim et al. (2026). *Quantifying micro- and nanoplastics in blood, urine, and soft tissues using optimized cascaded microfiltration and pyrolysis-GC/MS for exposomic investigations*. Environmental Pollution. https://doi.org/10.1016/j.envpol.2026.128553
- V. Almeida et al. (2025). *Application of a convolutional neural network for automated multiclass identification of field-collected microplastics and diatom algae from optical microscopy images*. Revista Brasileira de Ciências Ambientais. https://doi.org/10.5327/z2176-94782491
- Valentina Ferrari et al. (2026). *Optimizing microplastic extraction from organic-rich and bristle polychaetes (Hermodice carunculata)*. Water Biology and Security. https://doi.org/10.1016/j.watbs.2026.100621
- Valeria Hidalgo‐Ruz et al. (2012). *Microplastics in the Marine Environment: A Review of the Methods Used for Identification and Quantification*. Environmental Science & Technology. https://doi.org/10.1021/es2031505
- Vara Prasad Kasa et al. (2022). *Microplastics in soil: Current status and evaluation of the greenness of various analytical methods of identification*. Green Analytical Chemistry. https://doi.org/10.1016/j.greeac.2022.100038
- Veera Shakar Pulusu et al. (2025). *Microplastics in the Environment: Sources, Detection Techniques, and Analytical Challenges*. OALib. https://doi.org/10.4236/oalib.1113697
- Veronica Nava & Barbara Leoni (2021). *Comparison of Different Procedures for Separating Microplastics from Sediments*. Water. https://doi.org/10.3390/w13202854
- Vidhya Selvaraj et al. (2026). *Plastic-catch: A Paper-based fluorescence method for salivary microplastic screening*. Journal of Contemporary Orthodontics. https://doi.org/10.18231/j.jco.14217.1789120048
- Vittorio Bianco et al. (2019). *Microplastic Identification via Holographic Imaging and Machine Learning*. Advanced Intelligent Systems. https://doi.org/10.1002/aisy.201900153
- Vlatka Mikulec et al. (2023). *Green Techniques for Detecting Microplastics in Marine with Emphasis on FTIR and NIR Spectroscopy—Short Review*. Processes. https://doi.org/10.3390/pr11082360
- Xinyu Yan et al. (2022). *An Ensemble Machine Learning Method for Microplastics Identification with Ftir Spectrum*. Journal of environmental chemical engineering. https://doi.org/10.1016/j.jece.2022.108130
- Yan Zhang, Dan Zhang & Z. Zhang (2023). *A Critical Review on Artificial Intelligence—Based Microplastics Imaging Technology: Recent Advances, Hot-Spots and Challenges*. International Journal of Environmental Research and Public Health. https://doi.org/10.3390/ijerph20021150
- Yasemin Umurhan et al. (2025). *Applications of Raman spectroscopy for microplastic detection and characterization: a comprehensive spectral reference*. Environmental Science and Pollution Research. https://doi.org/10.1007/s11356-025-37224-3
- Yiannis Kiouvrekis, Ioannis Psomadakis & Theodor Panagiotakopoulos (2026). *A Systematic Review of Artificial Intelligence and Machine Learning Techniques for Microplastic Detection and Analysis*. Microplastics. https://doi.org/10.3390/microplastics5030165
- Yingying Zhong et al. (2022). *Analysis of Microplastics in Aquatic Shellfish by Pyrolysis–Gas Chromatography/Mass Spectrometry after Alkali Digestion and Solvent Extraction*. Polymers. https://doi.org/10.3390/polym14183888
- Yoshio Ikezawa (2025). *Automatic Sample Preparation Device proposed as one of homoginized methods to collect natural microplastics in environmental waters*. . https://doi.org/10.5194/oos2025-435
- Yuanli Liu, Gao‐Feng Zhao & Fan Liu (2025). *Deep Learning-Based Shape Classification for Hyperspectral-Imaged Microplastics*. Analytical Chemistry. https://doi.org/10.1021/acs.analchem.5c02683
- Yuxuan Hou et al. (2026). *Portable microwave sensor system for real-time detection of microplastics in seawater*. Microsystems & Nanoengineering. https://doi.org/10.1038/s41378-026-01413-y
- Zeynep Çelik & Irmak Yıldırım (2025). *Microplastic Contamination Assessment in Urban Stormwater Runoff Using Hyperspectral Imaging and Machine Learning: A Case Study of the İzmirMetropolitan Drainage Network, Turkey*. Aintelia Science Notes. https://doi.org/10.68099/asnj.2025.82
- Zhiqiang Gao, Kendall Wontor & James V. Cizdziel (2022). *Labeling Microplastics with Fluorescent Dyes for Detection, Recovery, and Degradation Experiments*. Molecules. https://doi.org/10.3390/molecules27217415
- В. С. Чулков et al. (2025). *Molecular and Cellular Effects of Microplastics and Nanoplastics in the Pathogenesis of Cardiovascular, Nervous, Urinary, Digestive, and Reproductive System Diseases: A Global Systematic Review*. International Journal of Molecular Sciences. https://doi.org/10.3390/ijms262211194

---

## Appendix A — Search strings and per-database hits

Full manifest with every query, database, retrieval date, reported hit count, retrieved count and status: `runs/litreview_v2/sources/manifest.json` (themed + anchors) and `runs/litreview_v2/sources/manifest_chain.json` (citation chaining).

### A.1 Themed queries

| Theme | Query |
|---|---|
| T1 digestion | microwave assisted digestion microplastics extraction recovery sediment |
| T1 digestion | acid digestion hydrogen peroxide microplastics sample preparation matrix |
| T2 staining | Nile Red staining microplastics fluorescence protocol recovery specificity |
| T2 staining | Nile Red microplastics false positive natural organic material selectivity |
| T3 image analysis | machine learning image analysis microplastics detection counting classification |
| T3 image analysis | deep learning segmentation microplastic particles fluorescence microscopy |
| T4 quantification/QA | microplastics quantification detection limit recovery blank sediment fish |
| T4 quantification/QA | microplastics interlaboratory comparison harmonization quality assurance quantification |
| T5 matrices | microplastics sediment soil fish tissue digestion quantification monitoring |

Each query was run against all five databases: 45 database-query combinations from 9 themed queries. Total search calls including the anchor retrievals and the citation-chaining stage: 57, of which 5 failed — 4 arXiv calls (HTTP 429 or read timeout) and the forward chain's first attempt (HTTP 400).

### A.2 Anchor and chaining queries

| Route | Query |
|---|---|
| Anchor A1 | doi:10.1371/journal.pone.0314520 |
| Anchor A2 | doi:10.4067/s0717-97072024000106082 |
| Forward chain A1 | OpenAlex `cites:W4405493624` |
| Backward chain A1 | OpenAlex `referenced_works` of W4405493624, plus the Crossref reference list |
| Forward chain A2 | OpenAlex `cites:W4404352343` |
| Backward chain A2 | OpenAlex `referenced_works` (empty for this record) and the Crossref reference list (empty) |

## Appendix B — PRISMA counts

Machine-readable: `runs/litreview_v2/prisma_v2.json`. The flow diagram in §3.1 is generated from that file, not written by hand.

## Appendix C — Extraction table

`runs/litreview_v2/extraction_table_v2.csv` — 287 rows, one per included study, with the pillar coding, evidence type, scope/matrix, method family, the first quantitative result recoverable from the abstract, four quality flags, the retrieval route and the extraction basis. Fields are rule-derived; see §2.5.

## Appendix D — Tooling contribution

- `runs/litreview_v2/search_v2.py` — themed multi-database search with per-call manifest
- `runs/litreview_v2/chain_v2.py` — citation chaining in both directions, with the OpenAlex work-ID correction and the Crossref reference-list fallback
- `runs/litreview_v2/screen_v2.py` — deduplication, deterministic tiered screening, PRISMA counts
- `runs/litreview_v2/fetch_v2.py` — per-study metadata and abstract retrieval
- `runs/litreview_v2/extract_v2.py` — rule-based extraction with the quality flags
- `runs/litreview_v2/sources/citation_report_v2.json` — DOI registration and metadata verification
