There's just been a bunch of issues. There's 3 steps to post processing:

1) generate fragments, they look like this:

Screenshot from 2023-06-12 13-50-58png

mostly good but lots of noise outside the brain. Here's a more zoomed in view:

Screenshot from 2023-06-12 13-54-27png

2) is to generate edges between fragments so we know which ones to merge together

3) is to run a global agglomeration to get final ids, ideally combine all the fragments that belong to the same cells so that we get rid of the block boundaries 
I have 2 approaches to solve all of these steps (mutex watershed vs default watershed), default is what I've used before and is super sensitive to merge errors. Because the newer predictions pick up more cells, the predictions are denser and there are more merge errors, leading to an agglomeration that looks like this (everything merged together), which isn't very helpful

This is why I've been looking into mutex watershed which can support negative edges, so you can avoid merging cells that have a small merge error if there is significant evidence that they are not the same object. This is very promising but was running into a lot of memory errors and taking over 24 hours to run each individual step. I ended up having to scrap that approach due to it just being way too slow.

I'm now trying my last idea which is a combo approach where we use the older method for step 1, and the new mutex method for step 2/3. This is looking super promising, getting good results in a few hours instead of multiple days, here's a side by side of a region that has finished so far

This is why l've been looking into mutex watershed which can support negative edges, so you can avoid merging cells that have a small merge error if there is significant evidence that they are not the same object. This is very promising but was running into a lot of memory errors and taking over 24 hours to run each individual step. I ended up having to scrap that approach due to it just being way too slow.

I'm now trying my last idea which is a combo approach where we use the older method for step 1, and the new mutex method for step 2/3. This is looking super promising, getting good results in a few hours instead of multiple days, here's a side by side of a region that has finished so far (edited)

Screenshot from 2023-06-12 14-03-37..

Screenshot from 2023-06-12 13-54-27.

Unfortunately janelia's cluster seems to only be letting me run 7 workers so its taking about 20 hours, where normally I would run 100-200 workers and finish it up in under an hour. I'm looking into whats going on there, but at this rate I think the final volume should be ready tomorrow
