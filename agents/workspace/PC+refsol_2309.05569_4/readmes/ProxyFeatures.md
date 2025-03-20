# Proxy features
This experiment shows how ITI-GEN uses gender as a proxy attribute when learning other attributes, such as "Bald", "Mustache" or "Beard". 

## Train ITI-GEN inclusive tokens
You have to run `jobfiles/proxy_features/iti_gen/generation/${attribute}.sh`. To run all of them
```shell
find jobfiles/proxy_features/iti_gen/train/ -name "*.sh" | xargs -n1 bash
```

## Generation
You have to run `jobfiles/proxy_features/${method}/generation/${attribute}.sh`. To run all of them
```shell
find jobfiles/proxy_features/*/generation -name *.sh | xargs -n1 bash
```

## Evaluation
### Compute KL-divergences
Using the default parameters, 104 images are generated per category.
You need to run `jobfiles/proxy_features/${method}/evaluation/${attribute_list}.sh`, for example `jobfiles/proxy_features/hps_negative/evaluation/Male_Bald.sh`. To run all of them, do the following
```shell
find jobfiles/proxy_features -name "*.sh" | grep evaluation | xargs -n1 bash
```
