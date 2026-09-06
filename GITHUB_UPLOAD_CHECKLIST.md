# GitHub upload checklist

Before making the repository public:

- [ ] Do not upload TT100K or CCTSDB data.
- [ ] Do not upload detector weights unless redistribution is permitted.
- [ ] Do not upload private absolute paths (`/home/...`, `/root/...`, `C:\\...`).
- [ ] Check `configs/datasets/*.yaml` contains only portable public paths.
- [ ] Decide on a license after checking dependencies and detector-repository licenses.
- [ ] If releasing trained generator checkpoints, state exactly which dataset/victim model each checkpoint uses.
- [ ] If reporting exact manuscript numbers in README, only claim numbers tied to verified experiment logs/checkpoints.
- [ ] Test one training command and one evaluation command in a clean environment before release.
