# DS-Project
Semester project for OEAS 805

My dissertation focuses on creating a paleohurricane reconstruction from a back-barrier pond in Newfoundland, Atlantic Canada, over the past 7000 years using sediment cores. This paleotempestological work commonly takes sediment cores and quantifies the grain size or coarse fraction by laser partical diffraction or sieving, and finds anomalous coarse peaks to define as event layers. This study specifically consists of 853 data points down the entire core, at 1cm intervals, with the percent coarse (exceeding 125um) noted. 

A paper published by Kelly McKeon in 2026 creates a novel approach to defining event history. Previously, this was done by finding anomalous coarse peaks, peaks exceeding a certain percentage of the CDF of the data, assigning them as events, and applying a 50-100 year sliding window to count the events in a given time period. This assumes no sampling or chronological uncertainty, assuming that every coarse peak is an event, and that the core recorded every event that occurred during the time scale of the core. It also assumes that the ages of each event are the median ages given by the age model. McKeon's approach to address this is to perform a bootstrapping analysis that accounts for sampling uncertainty by resampling event times with replacement, and accounts for chronological uncertainty by randomizing the age model iteration that is used. 

My project will adapt this code to my own data to assess event frequency and be able to compare to nearby and similar datasets. 
