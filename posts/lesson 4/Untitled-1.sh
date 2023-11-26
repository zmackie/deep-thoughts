#! /usr/bin/env bash

docker run --pull always -it -v /root/scrape/data-disk:/var/scrape --entrypoint /bin/bash --network selenium registry.digitalocean.com/brokernomics/brokerscraper:fuck-latest