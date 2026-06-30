## 2026-05-28

### $\omega$ annual cycle analysis and projections of TAM correlation anoms
Investigated annual cycles of $\omega$ at 300 and 850 hPa, projected onto the correlation anomalies of $\omega$(300) and (850) x TAM.
See notebooks/z850indexanalysis.ipynb

Added significance test function to calculate and plot stippling on non-sig contour areas on the cross-correlation plots. 

![w300_seasonal_projection](figs_anims/W300_corr_seasonal_mean.png)
![w850_seasonal_projection](figs_anims/W850_corr_seasonal_mean.png)

## 2025-06-09

### TEM pressure velocity $\omega$* analysis and correlation with TAM
Calculated the TEM framework pressure velocity and correlated with TAM for DJF and JJA.

See notebooks/temanalysis.ipynb

![w*300_seasonal_correlation](figs_anims/W*(300)_corr_seasonal.png)

## 2026-06-27

### Seasonal w300 lag correlation with TAM (all four seasons)

Extended seasonal analysis from DJF/JJA to all four seasons using corrected
`djf_jja_mam_son` function (fixed SON DOY off-by-one and year wrap with pad=40).

Visually, no notable change in structure across all seasons for omega at 300 hPa: 

![w300 seasonal correlation](figs_anims/w300_corr_seasonal.png)

Same analysis but on TEM omega at 300 hPa:

![wstar300 seasonal correlation](figs_anims/wstar300_corr_seasonal.png)
