# Data processing

Consult the [Data preparation workflow](https://code.cs.uky.edu/seales-research/ink-id/-/wikis/Data-preparation-workflow) for more information on these steps.

- [ ] Transfer dataset to machine
- [ ] Resize/crop slices
- [ ] Consider windowing slices or adjusting contrast
- [ ] Create .volpkg or volume in existing .volpkg
- [ ] Segment
- [ ] Texture
- [ ] Orient correctly
- [ ] Align reference image
- [ ] Create labels (ink mask and aligned image)
- [ ] Create region set for this dataset
- [ ] Run ML
- [ ] Adjust and repeat until content
- [ ] Back up processed data to DRI-Datasets
- [ ] Add to Herculaneum master region set file, if applicable
- [ ] Add to benchmark test suite, if applicable 
- [ ] Update [Moonshot Data Progress Tracking Sheet](https://docs.google.com/spreadsheets/d/16s8GkQ74w5fmp6d1MwYGtmcf26gk9PjrD_ldManLhKw/edit#gid=0)

/label ~data