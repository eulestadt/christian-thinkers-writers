# Four top-40 work lists

Generated 2026-10-31 from read-only [`intrinsic-motivation-scores.sqlite`](intrinsic-motivation-scores.sqlite) (not modified), treatment [`corpus_manifest.json`](../cai-pipeline/data/corpus_manifest.json), and control [`corpus_manifest.json`](../cai-pipeline-control/data/corpus_manifest.json) plus [`cai-control-coverage.md`](cai-control-coverage.md).

GitHub copies: [this file](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md), [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md), [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md).

The **github coverage link** column points at the matching work row on GitHub `main`. I/E-R treatment works and citation-control works use those CAI coverage pages (titles there link to Gutenberg catalog `pd_url`). Hoge / I−E works that are not in either CAI arm link to this file.

## Ranking rules

1. **I/E-R intrinsic** — `works_scored` ⋈ `work_subscores` where `subscale='ier_intrinsic'` `ORDER BY score DESC LIMIT 40`. Join `author_decisions.canonical_name`; year from `work_candidates.year`. Same 40 as the CAI treatment arm (no I−E filter). One scored English PD work per author.
2. **I minus E** — same join, `subscale='I_minus_E'` (stored composite intrinsic minus composite extrinsic; also on `works_scored.I_minus_E`). Not I/E-R intrinsic minus (Ep+Es).
3. **Hoge intrinsic** — same join, `subscale='hoge_intrinsic'`.
4. **Control 40** — citation-control arm as stored in control coverage / manifest (year window 1500–1930, treatment works and authors excluded). **Score** is OpenAlex `cited_by_count` when that is > 0 (ranks 1–9); otherwise Wikidata sitelinks (ranks 10–40; Wikipedia inlinks are capped at 500 for all of those rows).

Publication years include `author_death_latest` bounds when that is the stored `year_source`.

## 1. Top 40 I/E-R intrinsic

`subscale='ier_intrinsic'`. Score is the stored I/E-R intrinsic work score.

| rank | author | title | year | score | github coverage link |
|---|---|---|---|---|---|
| 1 | Brother Lawrence | <a id="ier-q3090506"></a>The Practice of the Presence of God | 1692 | 0.00656683 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L31) |
| 2 | William Henry Griffith Thomas | <a id="ier-q8012011"></a>The Prayers of St. Paul | 1914 | 0.00616133 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L32) |
| 3 | Edward McKendree Bounds | <a id="ier-q4018797"></a>Preacher and Prayer | 1907 | 0.00533195 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L33) |
| 4 | Lilian Staveley | <a id="ier-q6547707"></a>The Romance of the Soul | 1920 | 0.00437728 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L34) |
| 5 | Albertus Magnus | <a id="ier-q60059"></a>On Union with God | 1911 | 0.00347414 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L35) |
| 6 | George Adam Smith | <a id="ier-q5536061"></a>Four Psalms: XXIII, XXXVI, LII, CXXI, Interpreted for Practical Use | 1896 | 0.00301618 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L36) |
| 7 | Augustus Hopkins Strong | <a id="ier-q4821449"></a>Systematic Theology: The Doctrine of God (1907) | 1907 | 0.00296311 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L37) |
| 8 | Ignatius of Loyola | <a id="ier-q44281"></a>Spiritual Exercises | 1548 | 0.00254304 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L38) |
| 9 | Legh Richmond | <a id="ier-q6518053"></a>The Dairyman's Daughter | 1827 | 0.00228953 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L39) |
| 10 | Francis de Sales | <a id="ier-q231256"></a>Maxims and Counsels of St. Francis de Sales for Every Day in the Year | 1884 | 0.00221390 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L40) |
| 11 | Francis John McConnell | <a id="ier-q5481512"></a>Understanding the Scriptures | 1917 | 0.00219616 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L41) |
| 12 | Rufus Jones | <a id="ier-q2713471"></a>The Double Search: Studies in Atonement and Prayer | 1906 | 0.00216327 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L42) |
| 13 | Cynewulf | <a id="ier-q560915"></a>The Elene of Cynewulf | 1904 | 0.00197948 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L43) |
| 14 | Arthur Tappan Pierson | <a id="ier-q4800431"></a>George Muller of Bristol and his Witness to a Prayer-Hearing God | 1899 | 0.00197408 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L44) |
| 15 | Francis Thompson | <a id="ier-q367745"></a>The Hound of Heaven | 1893 | 0.00187751 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L45) |
| 16 | Rosalind Goforth | <a id="ier-q7367226"></a>How I Know God Answers Prayer | 1921 | 0.00182846 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L46) |
| 17 | Joseph Bates | <a id="ier-q1365446"></a>The Opening Heavens Or, A Connected View of the Testimony of the Prophets and Apostles (1846) | 1846 | 0.00179022 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L47) |
| 18 | Catherine Booth | <a id="ier-q431614"></a>Godliness (1881) | 1881 | 0.00176755 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L48) |
| 19 | Hastings Rashdall | <a id="ier-q4209439"></a>Philosophy and Religion | 1910 | 0.00167333 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L49) |
| 20 | John Taylor | <a id="ier-q540244"></a>The Government of God (1852) | 1852 | 0.00166878 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L50) |
| 21 | Adoniram Judson Gordon | <a id="ier-q4684528"></a>The Ministry of the Spirit | 1895 | 0.00165476 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L51) |
| 22 | Albert Benjamin Simpson | <a id="ier-q2895878"></a>Days of Heaven Upon Earth: A Year Book of Scripture Texts and Living Truths | 1897 | 0.00163270 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L52) |
| 23 | Mary Baker Eddy | <a id="ier-q235069"></a>Science and Health with Key to the Scriptures | 1875 | 0.00163189 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L53) |
| 24 | William Law | <a id="ier-q3177480"></a>The Supersensual Life | 1901 | 0.00157235 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L54) |
| 25 | William Blake | <a id="ier-q41513"></a>Songs of Innocence and of Experience | 1794 | 0.00154316 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L55) |
| 26 | Julian of Norwich | <a id="ier-q236699"></a>Revelations of Divine Love | 1500 | 0.00151306 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L56) |
| 27 | William Temple | <a id="ier-q336315"></a>Church and Nation | 1915 | 0.00147097 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L57) |
| 28 | Hudson Taylor | <a id="ier-q470718"></a>A Retrospect | 1894 | 0.00141046 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L58) |
| 29 | Alphonsus Maria de Liguori | <a id="ier-q273334"></a>The Glories of Mary | 1750 | 0.00140897 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L59) |
| 30 | Henry Wace | <a id="ier-q5729746"></a>The War and the Gospel: Sermons & Addresses During the Present War | 1917 | 0.00139980 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L60) |
| 31 | Kuno Meyer | <a id="ier-q76911"></a>Selections from Ancient Irish Poetry | 1911 | 0.00139774 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L61) |
| 32 | Hannah T. King | <a id="ier-q41505357"></a>Brigham Young | 1872 | 0.00138234 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L62) |
| 33 | John Gresham Machen | <a id="ier-q1700283"></a>The Literature and History of New Testament Times | 1915 | 0.00133723 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L63) |
| 34 | Alban Butler | <a id="ier-q1617267"></a>The Lives of the Fathers, Martyrs, and Other Principal Saints | 1773 | 0.00127607 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L64) |
| 35 | John Keble | <a id="ier-q955187"></a>The Christian Year | 1827 | 0.00126680 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L65) |
| 36 | Ellet J. Waggoner | <a id="ier-q5365272"></a>The Glad Tidings (1900 Original) | 1900 | 0.00126294 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L66) |
| 37 | Arthur Shearly Cripps | <a id="ier-q3624318"></a>Primavera: Poems by Four Authors | 1900 | 0.00124379 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L67) |
| 38 | Joseph F. Smith | <a id="ier-q982197"></a>Gospel Doctrine: Selections from the Sermons and Writings of Joseph F. Smith (1919) | 1919 | 0.00124032 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L68) |
| 39 | Adolf von Harnack | <a id="ier-q164699"></a>The History of Dogma | 1930 | 0.00124005 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L69) |
| 40 | Charles Gore | <a id="ier-q366142"></a>Lux Mundi | 1889 | 0.00123591 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L70) |

## 2. Top 40 I minus E

`subscale='I_minus_E'`. Score is the stored I−E contrast.

| rank | author | title | year | score | github coverage link |
|---|---|---|---|---|---|
| 1 | Mary Townsend | <a id="ime-q98933269"></a>The Anti-Slavery Alphabet | 1846 | 0.00516075 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L71) |
| 2 | Albertus Magnus | <a id="ime-q60059"></a>On Union with God | 1911 | 0.00389327 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L35) |
| 3 | J. R. R. Tolkien | <a id="ime-q892"></a>A Middle English Vocabulary | 1922 | 0.00382544 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L73) |
| 4 | Cotton Mather | <a id="ime-q380719"></a>Wonders of the Invisible World | 1692 | 0.00371262 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L74) |
| 5 | Robert Burns | <a id="ime-q81960"></a>Tam O'Shanter | 1790 | 0.00343875 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L75) |
| 6 | Siegfried Sassoon | <a id="ime-q363729"></a>Counter-Attack and Other Poems | 1918 | 0.00311450 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L76) |
| 7 | William Law | <a id="ime-q3177480"></a>The Supersensual Life | 1901 | 0.00288420 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L54) |
| 8 | Mary Baker Eddy | <a id="ime-q235069"></a>Science and Health with Key to the Scriptures | 1875 | 0.00274870 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L53) |
| 9 | Brother Lawrence | <a id="ime-q3090506"></a>The Practice of the Presence of God | 1692 | 0.00263298 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L31) |
| 10 | George Adam Smith | <a id="ime-q5536061"></a>Four Psalms: XXIII, XXXVI, LII, CXXI, Interpreted for Practical Use | 1896 | 0.00230736 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L36) |
| 11 | Léon Daudet | <a id="ime-q92562"></a>Alphonse Daudet | 1898 | 0.00225869 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L81) |
| 12 | Francis de Sales | <a id="ime-q231256"></a>Maxims and Counsels of St. Francis de Sales for Every Day in the Year | 1884 | 0.00225431 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L40) |
| 13 | Ellet J. Waggoner | <a id="ime-q5365272"></a>The Glad Tidings (1900 Original) | 1900 | 0.00219676 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L66) |
| 14 | Augustus Hopkins Strong | <a id="ime-q4821449"></a>Systematic Theology: The Doctrine of God (1907) | 1907 | 0.00218430 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L37) |
| 15 | John Calvin | <a id="ime-q37577"></a>Institutes of the Christian Religion | 1536 | 0.00217611 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L85) |
| 16 | Francis Thompson | <a id="ime-q367745"></a>The Hound of Heaven | 1893 | 0.00211859 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L45) |
| 17 | Lilias Trotter | <a id="ime-q6547787"></a>Parables of the Cross (1890) | 1890 | 0.00203117 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L87) |
| 18 | Samuel Taylor Coleridge | <a id="ime-q82409"></a>The Rime of the Ancient Mariner | 1798 | 0.00196963 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L88) |
| 19 | Pierre Corneille | <a id="ime-q747"></a>Polyeucte | 1642 | 0.00188370 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L89) |
| 20 | Joel Chandler Harris | <a id="ime-q554269"></a>Uncle Remus | 1908 | 0.00185754 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L90) |
| 21 | Gregory of Tours | <a id="ime-q67841"></a>The History of the Franks | 1927 | 0.00184295 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L91) |
| 22 | William Carlos Williams | <a id="ime-q178106"></a>Sour Grapes | 1921 | 0.00184142 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L92) |
| 23 | E. A. Wallis Budge | <a id="ime-q2316289"></a>The Book of the Dead: The Papyrus of Ani in the British Museum; the Egyptian Text with Interlinear Transliteration and Translation, a Running Translation, Introduction, Etc. | 1895 | 0.00183299 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L93) |
| 24 | Murray Leinster | <a id="ime-q550449"></a>The Runaway Skyscraper | 1919 | 0.00180568 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L94) |
| 25 | Ludwig Lavater | <a id="ime-q120784"></a>Of Ghostes and Spirites, Walking by Night | 1596 | 0.00172492 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L95) |
| 26 | Joseph Fielding Smith | <a id="ime-q1392595"></a>Blood Atonement and the Origin of Plural Marriage | 1905 | 0.00171671 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L96) |
| 27 | John Newton | <a id="ime-q357301"></a>Thoughts upon the African Slave Trade | 1788 | 0.00167763 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L97) |
| 28 | William Greenough Thayer Shedd | <a id="ime-q8010175"></a>Sermons to the Natural Man | 1871 | 0.00161181 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L98) |
| 29 | Erich Maria Remarque | <a id="ime-q47293"></a>All Quiet on the Western Front | 1928 | 0.00157220 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L99) |
| 30 | Grazia Deledda | <a id="ime-q7728"></a>After the Divorce | 1902 | 0.00154344 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L100) |
| 31 | Catherine Booth | <a id="ime-q431614"></a>Godliness (1881) | 1881 | 0.00154274 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L48) |
| 32 | Charles Spurgeon | <a id="ime-q278825"></a>Around the Wicket Gate | 1892 | 0.00149538 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L102) |
| 33 | Jan Kochanowski | <a id="ime-q59737"></a>Laments | 1580 | 0.00138384 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L103) |
| 34 | William Muir | <a id="ime-q1391199"></a>The Caliphate: Its Rise, Decline, and Fall | 1915 | 0.00135605 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L104) |
| 35 | Richard Baxter | <a id="ime-q680070"></a>A Christian Directory: Or, A Body of Practical Divinity and Cases of Conscience | 1825 | 0.00134609 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L105) |
| 36 | Robert Barclay | <a id="ime-q443578"></a>An Apology for the true Christian Divinity | 1676 | 0.00133540 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L106) |
| 37 | Ellen G. White | <a id="ime-q292290"></a>The Great Controversy | 1858 | 0.00129769 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L107) |
| 38 | Lewis Sperry Chafer | <a id="ime-q5973650"></a>Satan: His Motive and Methods | 1919 | 0.00126244 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L108) |
| 39 | Rosalind Goforth | <a id="ime-q7367226"></a>How I Know God Answers Prayer | 1921 | 0.00123715 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L46) |
| 40 | Ezra Pound | <a id="ime-q163366"></a>Hugh Selwyn Mauberley | 1920 | 0.00122527 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L110) |

## 3. Top 40 Hoge intrinsic

`subscale='hoge_intrinsic'`. Score is the stored Hoge 1972 IRMS intrinsic work score.

| rank | author | title | year | score | github coverage link |
|---|---|---|---|---|---|
| 1 | William Henry Griffith Thomas | <a id="hoge-q8012011"></a>The Prayers of St. Paul | 1914 | 0.01726780 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L32) |
| 2 | Brother Lawrence | <a id="hoge-q3090506"></a>The Practice of the Presence of God | 1692 | 0.01367370 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L31) |
| 3 | Albertus Magnus | <a id="hoge-q60059"></a>On Union with God | 1911 | 0.01244698 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L35) |
| 4 | Francis de Sales | <a id="hoge-q231256"></a>Maxims and Counsels of St. Francis de Sales for Every Day in the Year | 1884 | 0.01107211 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L40) |
| 5 | Ignatius of Loyola | <a id="hoge-q44281"></a>Spiritual Exercises | 1548 | 0.00956193 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L38) |
| 6 | Edward McKendree Bounds | <a id="hoge-q4018797"></a>Preacher and Prayer | 1907 | 0.00925176 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L33) |
| 7 | Lilian Staveley | <a id="hoge-q6547707"></a>The Romance of the Soul | 1920 | 0.00909667 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L34) |
| 8 | Julian of Norwich | <a id="hoge-q236699"></a>Revelations of Divine Love | 1500 | 0.00900802 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L56) |
| 9 | Harry Emerson Fosdick | <a id="hoge-q1586481"></a>The Meaning of Faith | 1917 | 0.00895980 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L126) |
| 10 | Albert Benjamin Simpson | <a id="hoge-q2895878"></a>Days of Heaven Upon Earth: A Year Book of Scripture Texts and Living Truths | 1897 | 0.00893009 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L52) |
| 11 | Philipp Melanchthon | <a id="hoge-q76325"></a>Augsburg Confession | 1530 | 0.00812581 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L128) |
| 12 | Girolamo Savonarola | <a id="hoge-q164823"></a>Triumph of the Cross | 1498 | 0.00774744 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L129) |
| 13 | Augustus Hopkins Strong | <a id="hoge-q4821449"></a>Systematic Theology: The Doctrine of God (1907) | 1907 | 0.00731203 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L37) |
| 14 | Legh Richmond | <a id="hoge-q6518053"></a>The Dairyman's Daughter | 1827 | 0.00692947 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L39) |
| 15 | Arthur Tappan Pierson | <a id="hoge-q4800431"></a>George Muller of Bristol and his Witness to a Prayer-Hearing God | 1899 | 0.00650172 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L44) |
| 16 | Thomas Rogers | <a id="hoge-q7793621"></a>Thomas à Kempis | 1616 | 0.00639043 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L133) |
| 17 | Ellet J. Waggoner | <a id="hoge-q5365272"></a>The Glad Tidings (1900 Original) | 1900 | 0.00633640 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L66) |
| 18 | William Law | <a id="hoge-q3177480"></a>The Supersensual Life | 1901 | 0.00612330 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L54) |
| 19 | Mary Baker Eddy | <a id="hoge-q235069"></a>Science and Health with Key to the Scriptures | 1875 | 0.00611288 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L53) |
| 20 | Dwight Lyman Moody | <a id="hoge-q453218"></a>Secret Power | 1881 | 0.00603270 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L137) |
| 21 | Alphonsus Maria de Liguori | <a id="hoge-q273334"></a>The Glories of Mary | 1750 | 0.00569011 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L59) |
| 22 | Thomas à Kempis | <a id="hoge-q220976"></a>The Imitation of Christ | 1471 | 0.00561347 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L139) |
| 23 | Catherine Booth | <a id="hoge-q431614"></a>Godliness (1881) | 1881 | 0.00554684 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L48) |
| 24 | Hannah T. King | <a id="hoge-q41505357"></a>Brigham Young | 1872 | 0.00531822 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L62) |
| 25 | George Adam Smith | <a id="hoge-q5536061"></a>Four Psalms: XXIII, XXXVI, LII, CXXI, Interpreted for Practical Use | 1896 | 0.00528468 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L36) |
| 26 | Henry Wace | <a id="hoge-q5729746"></a>The War and the Gospel: Sermons & Addresses During the Present War | 1917 | 0.00525691 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L60) |
| 27 | Adoniram Judson Gordon | <a id="hoge-q4684528"></a>The Ministry of the Spirit | 1895 | 0.00520353 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L51) |
| 28 | Joseph F. Smith | <a id="hoge-q982197"></a>Gospel Doctrine: Selections from the Sermons and Writings of Joseph F. Smith (1919) | 1919 | 0.00514166 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L68) |
| 29 | Rufus Jones | <a id="hoge-q2713471"></a>The Double Search: Studies in Atonement and Prayer | 1906 | 0.00512072 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L42) |
| 30 | Rebecca Ruter Springer | <a id="hoge-q7301895"></a>Intra Muros | 1904 | 0.00478616 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L147) |
| 31 | Francis John McConnell | <a id="hoge-q5481512"></a>Understanding the Scriptures | 1917 | 0.00472526 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L41) |
| 32 | Samuel Dickey Gordon | <a id="hoge-q24284391"></a>Quiet Talks on Power (1903) | 1903 | 0.00470438 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L149) |
| 33 | John Calvin | <a id="hoge-q37577"></a>Institutes of the Christian Religion | 1536 | 0.00468122 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L150) |
| 34 | Thieleman J. van Braght | <a id="hoge-q7784006"></a>Martyrs Mirror | 1660 | 0.00464655 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L151) |
| 35 | Charles Spurgeon | <a id="hoge-q278825"></a>Around the Wicket Gate | 1892 | 0.00463419 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L152) |
| 36 | Henry Drummond | <a id="hoge-q328513"></a>The Greatest Thing in the World: an Address | 1890 | 0.00453306 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L153) |
| 37 | Walter Elliott | <a id="hoge-q7964762"></a>Life of Father Hecker | 1891 | 0.00440444 | [this list](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/four-top40-lists.md#L154) |
| 38 | Joseph Bates | <a id="hoge-q1365446"></a>The Opening Heavens Or, A Connected View of the Testimony of the Prophets and Apostles (1846) | 1846 | 0.00427910 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L47) |
| 39 | Francis Thompson | <a id="hoge-q367745"></a>The Hound of Heaven | 1893 | 0.00424962 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L45) |
| 40 | Charles Gore | <a id="hoge-q366142"></a>Lux Mundi | 1889 | 0.00418471 | [treatment coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-deepseek-coverage.md#L70) |

## 4. Control 40

Citation-control CAI arm. Score as defined under Ranking rules.

| rank | author | title | year | score | github coverage link |
|---|---|---|---|---|---|
| 1 | William James | <a id="ctrl-pg57628"></a>The Principles of Psychology | 1890 | 25773 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L33) |
| 2 | Charles Darwin | <a id="ctrl-pg1227"></a>The Expression of the Emotions in Man and Animals | 1872 | 12119 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L34) |
| 3 | Sigmund Freud | <a id="ctrl-pg66048"></a>The Interpretation of Dreams | 1913 | 6957 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L35) |
| 4 | John Dewey | <a id="ctrl-pg37423"></a>How We Think | 1910 | 6767 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L36) |
| 5 | Francis Galton | <a id="ctrl-pg11562"></a>Inquiries into human faculty and its development | 1883 | 2257 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L37) |
| 6 | D’Arcy Wentworth Thompson | <a id="ctrl-pg55264"></a>On Growth and Form | 1917 | 1917 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L38) |
| 7 | John Stuart Mill | <a id="ctrl-pg5669"></a>Considerations on Representative Government | 1862 | 1679 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L39) |
| 8 | Friedrich Nietzsche | <a id="ctrl-pg52914"></a>The Will to Power | 1901 | 1497 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L40) |
| 9 | Robert Ridgway | <a id="ctrl-pg63087"></a>Color Standards and Color Nomenclature | 1912 | 1473 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L41) |
| 10 | William Shakespeare | <a id="ctrl-q41567"></a>Hamlet | 1603 | 143 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L42) |
| 11 | Friedrich Engels | <a id="ctrl-q40591"></a>The Communist Manifesto | 1848 | 143 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L43) |
| 12 | Joseph Smith | <a id="ctrl-q459842"></a>Book of Mormon | 1830 | 126 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L44) |
| 13 | Fyodor Dostoyevsky | <a id="ctrl-q165318"></a>Crime and Punishment | 1866 | 109 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L45) |
| 14 | Niccolò Machiavelli | <a id="ctrl-q131719"></a>The Prince | 1532 | 101 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L46) |
| 15 | Jane Austen | <a id="ctrl-q170583"></a>Pride and Prejudice | 1813 | 100 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L47) |
| 16 | Daniel Defoe | <a id="ctrl-q483034"></a>Robinson Crusoe | 1719 | 99 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L48) |
| 17 | Victor Hugo | <a id="ctrl-q180736"></a>Les Misérables | 1862 | 98 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L49) |
| 18 | Alexandre Dumas | <a id="ctrl-q140527"></a>The Three Musketeers | 1844 | 88 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L50) |
| 19 | Johann Wolfgang von Goethe | <a id="ctrl-q29478"></a>Faust | 1808 | 85 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L51) |
| 20 | Robert Louis Stevenson | <a id="ctrl-q185118"></a>Treasure Island | 1883 | 85 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L52) |
| 21 | Franz Kafka | <a id="ctrl-q184222"></a>The Metamorphosis | 1915 | 84 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L53) |
| 22 | James Joyce | <a id="ctrl-q6511"></a>Ulysses | 1922 | 84 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L54) |
| 23 | F. Scott Fitzgerald | <a id="ctrl-q214371"></a>The Great Gatsby | 1925 | 84 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L55) |
| 24 | Charles Dickens | <a id="ctrl-q164974"></a>Oliver Twist | 1839 | 83 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L56) |
| 25 | Oscar Wilde | <a id="ctrl-q82464"></a>The Picture of Dorian Gray | 1890 | 82 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L57) |
| 26 | Martin Luther | <a id="ctrl-q157506"></a>Ninety-five Theses | 1517 | 76 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L58) |
| 27 | Charlotte Brontë | <a id="ctrl-q182961"></a>Jane Eyre | 1847 | 70 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L59) |
| 28 | Rudyard Kipling | <a id="ctrl-q189509"></a>The Jungle Book | 1894 | 69 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L60) |
| 29 | L. Frank Baum | <a id="ctrl-q130295"></a>The Wonderful Wizard of Oz | 1900 | 67 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L61) |
| 30 | Lewis Carroll | <a id="ctrl-q220331"></a>Through the Looking-Glass | 1871 | 61 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L62) |
| 31 | Thomas Hobbes | <a id="ctrl-q193034"></a>Leviathan | 1651 | 59 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L63) |
| 32 | Louisa May Alcott | <a id="ctrl-q523076"></a>Little Women | 1868 | 52 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L64) |
| 33 | Henrik Ibsen | <a id="ctrl-q669694"></a>A Doll's House | 1879 | 52 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L65) |
| 34 | Walter Scott | <a id="ctrl-q840974"></a>Ivanhoe | 1819 | 51 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L66) |
| 35 | George Bernard Shaw | <a id="ctrl-q637200"></a>Pygmalion | 1913 | 43 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L67) |
| 36 | Anton Chekhov | <a id="ctrl-q650063"></a>The Seagull | 1896 | 42 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L68) |
| 37 | Alexander Hamilton | <a id="ctrl-q858036"></a>Federalist Papers | 1788 | 41 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L69) |
| 38 | T. S. Eliot | <a id="ctrl-q581458"></a>The Waste Land | 1922 | 37 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L70) |
| 39 | J. M. Barrie | <a id="ctrl-q3435337"></a>Peter Pan | 1911 | 35 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L71) |
| 40 | John Locke | <a id="ctrl-q231949"></a>Two Treatises of Government | 1690 | 32 | [control coverage](https://github.com/eulestadt/christian-thinkers-writers/blob/main/docs/cai-control-coverage.md#L72) |
