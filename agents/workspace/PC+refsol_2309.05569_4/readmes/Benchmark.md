## Benchmark
This experiments shows that the training time is proportional to the size of the joint distribution, that grows exponentially with the number of attributes. You have to run `jobfiles/benchmark/${num_attributes}.sh` where `num_attributes` is a number between 1 and 8. To run all of them, do the following
```shell
find jobfiles/benchmark/*.sh | xargs -n1 bash
```