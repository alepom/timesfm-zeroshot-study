# references.bib: verification notes

All checks were done on 2026-10-30. Each line gives the key, the primary source used, and any caveat. 56 entries.

## 1. TimesFM
- `das2024timesfm`: PMLR v235 page https://proceedings.mlr.press/v235/das24c.html (citation meta: authors, pp. 10148-10167, 2024-07-08) and arXiv:2310.10688 (API). OK.
- `jain2026timesfm3`: Google Research Blog post https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/, dated **August 31, 2026**, byline "Ayush Jain and Rajat Sen, Research Scientists, Google Research". The GitHub README also links it. **No TimesFM-3 paper or technical report exists** (arXiv API search for "TimesFM-3"/"TimesFM 3" and a web search found none), so this blog post and the HF card are the primary sources. The HF 3.0 card's "Citation" still points to the 2023/2024 TimesFM paper. The blog's "Contiguous Patch Masking" link goes to arXiv:2505.23719, which is the TiRex paper (`auer2025tirex`).
- `timesfm2026github`: GitHub API + README https://github.com/google-research/timesfm (Apache-2.0, created 2024-04-29). The README dates the "TimesFM 3.0 is out" update to August 2026. Year = access year.
- `timesfm2025hf`: HF API + README https://huggingface.co/google/timesfm-2.5-200m-pytorch (created 2025-09-02, Apache-2.0). OK.
- `timesfm2026hf`: HF API + README https://huggingface.co/google/timesfm-3.0-pytorch (created 2026-08-24, last modified 2026-09-02, licence `timesfm-non-commercial-license-v1.0`). The card states the pretraining data: GiftEvalPretrain excluding fev-bench overlaps, Wikipedia pageviews to Nov 2023, Google Trends to end of 2022, synthetic data. OK.

## 2. Chronos
- `ansari2024chronos`: OpenReview (API) gerNCVqqtR, "Accepted by TMLR", published 2024-11-11, Expert Certification. Authors taken from arXiv:2403.07815 and the OpenReview bibtex. The GitHub README bibtex leaves out "Hao Wang", but arXiv and OpenReview both list him, so he is included. OK.
- `ansari2025chronos2`: arXiv:2510.15821 (API, 2025-10-17, 23 authors). OK.
- `chronos2026github`: GitHub API https://github.com/amazon-science/chronos-forecasting (Apache-2.0). Year = access year.

## 3. TiRex
- `auer2025tirex`: arXiv:2505.23719 (comment "presented at NeurIPS 2025") and OpenReview v7UqniC9pF ("NeurIPS 2025 poster"). The booktitle follows the NX-AI/tirex README. Volume and pages of the NeurIPS proceedings were **not** verified, so they are omitted.
- `podest2026tirex2`: arXiv:2607.01204 (API, 2026-07-01, 10 authors) and the NX-AI/tirex-2 README citation. arXiv only, no venue.
- `tirex2026hf`: HF API + README https://huggingface.co/NX-AI/TiRex-2 (created 2026-06-16, Apache-2.0). Author is given as the organisation "NXAI" (HF org NX-AI).

## 4. Benchmarks
- `shchur2025fevbench`: arXiv:2509.26468 (API, v4 2026-06-30, no journal-ref) and the autogluon/fev README citation. arXiv only.
- `fev2026github`: GitHub API https://github.com/autogluon/fev ("Forecast evaluation library", Apache-2.0). Year = access year.
- `aksu2024gifteval`: arXiv:2410.10393 (API) and the SalesforceAIResearch/gift-eval README. OpenReview Z2cMOOANFX confirms the NeurIPS 2024 TSALM workshop, which is mentioned in the note.
- `qiao2026time` (extra: TIME benchmark, which the TimesFM-3 README cites): arXiv:2602.12147 (comment "Accepted to ICML 2026. Camera-ready") and OpenReview 79TgfXHbsK ("ICML 2026 regular"). PMLR volume and pages are not yet verified and are omitted. The "43rd ICML" numbering comes from the Pan & Ezzat journal-ref (ICML 2026, Seoul).

## 5. Simulation study (arXiv:2610.00589)
- `ebrahim2026foundation`: arXiv API, exact title "Foundation or Formula? A Simulation-Based Comparison of Google's TimesFM-3 and Classical Time-Series Models", authors Ebrahim Khaled Ebrahim, Somaia Mohamed Ali, Ahmed El-Kotory. Submitted 2026-09-30, primary category stat.ME. OK.
- `ebrahim2026zenodo`: Zenodo API https://zenodo.org/api/records/22976253, Software, version 1.2.2, published 2026-09-26, same three authors. **Note:** the arXiv comment cites https://doi.org/10.5281/zenodo.22681072. That is the *concept* DOI (all versions) of the same record, and 22976253 is the version DOI (v1.2.2). Both resolve to the same compendium. Use the concept DOI instead if you want "latest version" semantics.

## 6. Other TSFMs
- `woo2024moirai`: PMLR v235 woo24a (pp. 53140-53164) and arXiv:2402.02592. OK.
- `liu2025moirai2`: arXiv:2511.11698 (API, v3 2026-02-03). No venue found. HF Salesforce/moirai-2.0-R-small tags this arXiv id. arXiv only.
- `rasul2023lagllama`: arXiv:2310.08278 (API, v1 2023-10-12, v3 2024-02-08). OpenReview shows a NeurIPS 2023 R0-FoMo workshop poster under a shorter title, so the entry cites arXiv only.
- `shi2025timemoe`: OpenReview e1wDDFmlVu ("ICLR 2025 Spotlight") and arXiv:2409.16040. OK.
- `liu2025sundial`: PMLR v267 index (liu25be, pp. 39295-39317, ICML 2025) and arXiv:2502.00816. OK.
- `cohen2025toto`: arXiv:2505.14766 (API, 19 authors) and the DataDog/toto README. This is the Toto (1.0) paper. arXiv only.
- `khwaja2026toto2` (extra: the TimesFM-3 blog benchmarks against Toto 2.0): arXiv:2605.20119 (API) and the DataDog/toto README. **Flag:** arXiv lists the author as "Viktoriya Zhukov" (the Toto 1.0 paper has "Zhukova"). The name is kept exactly as arXiv gives it.
- `hoo2026tabpfnts`: OpenReview (API) KIkQj8VOUY, "Accepted by TMLR", published 2026-07-07, official bibtex year 2026, and arXiv:2501.02945. OK.
- `goswami2024moment`: PMLR v235 goswami24a (pp. 16115-16152) and arXiv:2402.03885. OK.

## 7. Leakage / contamination
- `meyer2025leakage`: arXiv:2510.13654 (API, v1 2025-10-15, v3 2026-02-25). No venue found. It reports that only about 6% of 401 datasets used across TSFMs never appeared in a pretraining corpus, and it flags a possible TimesFM train/test overlap. A ResearchGate listing carries an earlier title, "Time Series Foundation Models: Benchmarking Challenges and Requirements", which appears to be v1.
- `meyer2026tsarena`: Crossref DOI 10.1145/3770855.3817515 (KDD '26 V.2, pp. 9558-9568, 5 authors) and arXiv:2512.20761. OK.
- `li2026tsfmaudit`: arXiv:2605.26161 (API, 2026-05-24, 10 authors). OpenReview shows it only as an ICLR 2027 *submission*, so it is cited as arXiv.
- `garza2026impermanent`: arXiv:2603.08707 (API) and OpenReview EOoinHRJ0P ("ICLR 2026 TSALM Workshop Poster"). Author names follow arXiv. OpenReview spells two authors "Renee Rosillo Garcia" and "Rodrigo Mendoza Smith".
- `pan2026electricity`: arXiv:2607.02623 (API; journal-ref "ICML 2026 Foundation Models for Structured Data Workshop"). OK.
- `hewamalage2023forecast` (extra: forecast-evaluation pitfalls including leakage): Crossref DOI 10.1007/s10618-022-00894-5, DMKD 37(2):788-832. Crossref "issued" is 2022-12-02 (online first), and the volume/issue is 2023, so year = 2023.
- Other search results were checked and left out as not specifically about leakage: SEER, MACROCAST, QuitoBench, TS-ICL. The "Time-Series Foundation Models That Understand Data Revisions" paper was also not added.

## 8. Classical methods / evaluation
- `hyndman2006mase`: Crossref 10.1016/j.ijforecast.2006.03.001. OK.
- `hyndman2008forecast`: Crossref 10.18637/jss.v027.i03 and the JSS article page meta (pp. 1-22, 2008-07-29). OK.
- `hyndman2002state`: Crossref 10.1016/S0169-2070(01)00110-8 (Hyndman, Koehler, Snyder, Grose). OK.
- `assimakopoulos2000theta`: Crossref 10.1016/S0169-2070(00)00066-2. OK.
- `bandara2025mstl`: Crossref 10.1504/IJOR.2025.143957 (Int. J. Operational Research 52(1):79-98, 2025, Inderscience) and arXiv:2107.13462 (2021). OpenAlex also shows an earlier 2022 IJOR "online first" DOI, 10.1504/ijor.2022.10048281, which is not used.
- `garza2022statsforecast`: Nixtla/statsforecast README "Citing" block (authors Azul Garza, Max Mergenthaler Canseco, Cristian Challú, Kin G. Olivares; PyCon Salt Lake City 2022) and the GitHub API. OK.
- `makridakis2020m4`: Crossref 10.1016/j.ijforecast.2019.04.014. OK.
- `makridakis2022m5`: Crossref 10.1016/j.ijforecast.2021.11.013 ("M5 accuracy competition: Results, findings, and conclusions"). OK.
- `gneiting2007scoring`: Crossref 10.1198/016214506000001437. OK.
- `gneiting2007calibration`: Crossref 10.1111/j.1467-9868.2007.00587.x. OK.

## 9. Statistics
- `holm1979`: **Partly verified.** There is no DOI (10.2307/4615733 returns 404 at Crossref/doi.org). JSTOR stable/4615733 could not be fetched (bot check). Metadata (title, Sture Holm, Scand. J. Stat. vol. 6, pp. 65-70, 1979) comes from OpenAlex, which links to JSTOR. The issue number is not confirmed by any fetched source (OpenAlex has issue = null), so `number` is omitted.
- `schuirmann1987tost`: Crossref 10.1007/BF01068419. OK.
- `cameron2011multiway`: Crossref 10.1198/jbes.2010.07136 ("Robust Inference With Multiway Clustering"). OK.
- `davison1997bootstrap`: Crossref 10.1017/CBO9780511802843 (monograph, CUP, 1997). The series name was not verified and is omitted.
- `wilson1927`: Crossref 10.1080/01621459.1927.10502953. OK.

## 10. Data sources
- `energycharts2026`: https://api.energy-charts.info/ (title "Energy-Charts API") and its openapi.json, which names Fraunhofer ISE, Freiburg, and gives CC BY 4.0 with "attribution: energy-charts.info" for most series. Some zones are CC BY 4.0 from Bundesnetzagentur | SMARD. Check the `license` field for the series you used.
- `zippenfenig2023openmeteo`: the citation Open-Meteo recommends on https://open-meteo.com/en/docs/historical-weather-api: "Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649". Zenodo API confirms 7970649 as the concept DOI. The same page also asks users to cite ERA5 (Hersbach et al.).
- `hersbach2020era5`: Crossref 10.1002/qj.3803 (QJRMS 146(730):1999-2049, all 43 authors). OK.
- `usgs2026nwis`: DataCite 10.5066/F7P55KJN ("USGS Water Data for the Nation", U.S. Geological Survey, URL waterdata.usgs.gov/nwis) and https://api.waterdata.usgs.gov/ ("USGS Water Data APIs"). **Flag:** DataCite gives 1994 as the issue year. The entry uses the access year (2026), as USGS's data-citation convention does. Adjust if you prefer.
- `ecb2026dataportal`: https://data.ecb.europa.eu/ ("ECB Data Portal") and https://data.ecb.europa.eu/help/api/overview. OK.
- `fred2026`: https://fred.stlouisfed.org/ ("Federal Reserve Economic Data | FRED | St. Louis Fed"). OK.
- `wikimedia2026pageviews`: https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html (title "Page view analytics | Wikimedia Analytics API"). The pageviews REST endpoint was tested live. Author "Wikimedia Foundation" is the host organisation, inferred, not stated as an author.
- `kaggle2017webtraffic`: the competition page title "Web Traffic Time Series Forecasting | Kaggle" was verified. The year 2017 is confirmed by the NIPS 2017 Time Series Workshop talk "Kaggle web traffic time series forecasting competition: results and insights" (V. Kuznetsov, https://neurips.cc/virtual/2017/10624). **Flag:** Kaggle's own "Citation" tab (personal author list) could not be read because the page is JS-only and the API needs authentication, so the author is given as "Kaggle".

## 11. Author's previous preprint
- `pomaro2026decision`: Zenodo API https://zenodo.org/api/records/23652645. Title matches, creator "Pomaro, Alessio", published 2026-10-26, Preprint, version 1.0.0, "Preprint, not peer reviewed". The concept DOI is 10.5281/zenodo.23652644. OK.

## UNVERIFIED / not included
- **TimesFM-3 paper or technical report**: none exists as of 2026-10-30 (arXiv and web search). The blog post and HF card are cited instead.
- **Kaggle Web Traffic competition personal author list**: not verifiable, so the organisation is used as author (entry included).
- Holm 1979 issue number and Davison & Hinkley series name: not confirmed (see above). Every other entry was verified in full.
