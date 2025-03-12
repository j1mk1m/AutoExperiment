# Reference Size
This experiment varies the number of reference images per category to prove that ITI-GEN is robust in the low-data regime. 
## Train
You need to run `jobfiles/reference_size/train/${attribute}_${reference_size}`, for example `jobfiles/reference_size/train/Age_10.sh`. To run all of them, do the following:
```shell
find jobfiles/reference_size/train/*.sh | xargs -n1 bash
```
## Generation
You need to run `jobfiles/reference_size/generation/${attribute}-${reference_size}.sh`, for example `jobfiles/reference_size/generation/Age-10.sh`. To run all of them, do the followin:
```shell
find jobfiles/reference_size/generation/*.sh | xargs -n1 bash
```