Artifact release for the paper "Duumviri: Detecting Trackers and Mixed Trackers with a Breakage Detector" published at NDSS 2025.


# Goals
Our artifacts should enable an user to 1. detect new non-mixed trackers, 2. detect new mixed trackers 3. replicate our evaluation on EasyList and EasyPrivacy and 4. inspect our mixed tracker evaluation dataset

# Enviroment

If you like to build your own enviroment. 
```bash
./docker_commands/build_docker.sh
```

Otherwise, you may use our Docker image with pre-configured enviroment. 

```bash
# To pull the image
docker pull 8759s/duumviri_no_brave_source:latest

# To run it
docker run -it --device /dev/fuse --privileged 8759s/duumviri_no_brave_source /bin/bash
```



# Detecting Non-mixed Trackers 
Input: a url of the page that you wish to find non-mixed trackers 

Output: requests that Duumviri analyzed and the analysis result (i.e., non-mixed tracker or not)

Command: 
```bash 
./detect_non_mixed_tracker.sh [URL]
```

## Example
```bash 
time ./detect_non_mixed_tracker.sh https://sec-deadlines.github.io/
```
On this site, Duumviri performs analysis on 16 requests and will print out the following output:
```
https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js False
https://platform.twitter.com/widgets.js False
https://cdnjs.cloudflare.com/ajax/libs/jquery.countdown/2.2.0/jquery.countdown.min.js False
https://cdnjs.cloudflare.com/ajax/libs/moment-timezone/0.5.34/moment-timezone-with-data-10-year-range.min.js False
https://www.google-analytics.com/g/collect?v=2&tid=G-JQ77F5GDMC&gtm=45je4730v9111283500za200&_p=1720146261922&gcd=13l3l3l3l2&npa=0&dma=0&tag_exp=95250752&ul=en-us&sr=800x600&cid=1954840087.1720146262&ir=1&uaa=x86&uab=64&uafvl=Chromium%3B111.0.5563.65%7CNot(A%253ABrand%3B8.0.0.0&uamb=0&uam=&uap=Linux&uapv=5.15.0&uaw=0&frm=0&pscdl=noapi&_eu=EBAI&_s=1&dl=https%3A%2F%2Fsec-deadlines.github.io%2F&dt=Security%20and%20Privacy%20Conference%20Deadlines&sid=1720146262&sct=1&seg=0&en=page_view&_fv=1&_ss=1&_ee=1&tfd=4762&_z=fetch False
https://www.google-analytics.com/collect?v=1&tid=UA-212479779-1&cid=777966b4-d241-440d-8bf3-d0b32a8a98bb&t=pageview&dp=%2Fbackground&dt=background&dh=chrome-extension%3A%2F%2Fapgipdcddhpolaocmabocdkahbbkggpm False
https://syndication.twitter.com/settings?session_id=c825ad404f6e56b2cd2a9edfdf0ac2e1d0c93eb2 False
https://platform.twitter.com/widgets/widget_iframe.2f70fb173b9000da126c79afe2098f02.html?origin=https%3A%2F%2Fsec-deadlines.github.io False
https://www.google-analytics.com/j/collect?v=1&_v=j101&a=289356269&t=pageview&_s=1&dl=https%3A%2F%2Fsec-deadlines.github.io%2F&ul=en-us&de=UTF-8&dt=Security%20and%20Privacy%20Conference%20Deadlines&sd=24-bit&sr=800x600&vp=765x476&je=0&_u=IEBAAEABAAAAACAAI~&jid=376111649&gjid=2141714929&cid=1954840087.1720146262&tid=UA-104921371-1&_gid=333907945.1720146262&_r=1&_slc=1&z=856167883 True
https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.29.1/moment.min.js False
https://platform.twitter.com/js/button.856debeac157d9669cf51e73a08fbc93.js False
https://cdnjs.cloudflare.com/ajax/libs/store.js/2.0.12/store.legacy.min.js False
https://sec-deadlines.github.io/static/js/main.js False
https://www.googletagmanager.com/gtag/js?id=G-JQ77F5GDMC&cx=c&_slc=1 True
https://syndication.twitter.com/i/jot/embeds?l=%7B%22widget_origin%22%3A%22https%3A%2F%2Fsec-deadlines.github.io%2F%22%2C%22widget_frame%22%3Afalse%2C%22language%22%3A%22en%22%2C%22message%22%3A%22m%3Anocount%3A%22%2C%22_category_%22%3A%22tfw_client_event%22%2C%22triggered_on%22%3A1720146262167%2C%22dnt%22%3Afalse%2C%22client_version%22%3A%222615f7e52b7e0%3A1702314776716%22%2C%22format_version%22%3A1%2C%22event_namespace%22%3A%7B%22client%22%3A%22tfw%22%2C%22page%22%3A%22button%22%2C%22section%22%3A%22share%22%2C%22action%22%3A%22impression%22%7D%7D&session_id=c825ad404f6e56b2cd2a9edfdf0ac2e1d0c93eb2 False
https://www.google-analytics.com/analytics.js True
```
This output is one request and analysis result per row. Taking the first two rows as an example, Duumrivir deems that `https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js` is not a non-mixed tracker, same with 
`https://platform.twitter.com/widgets.js`.


This example site took 87 mins, using a single 2.6 GHz core.

# Detecting Mixed Trackers 
Input: a url of the page that you wish to find mixed trackers 

Output: request, request field and value tuples that Duumviri analyzed and the analysis result (i.e., mixed tracker or not)

Command:
```bash 
./detect_mixed_tracker.sh [URL]
```

## Example
```bash 
time ./detect_mixed_tracker.sh https://sec-deadlines.github.io/
```

On this site, Duumviri performs analysis on 2 request fields and will print out the following output:
```python
('https://platform.twitter.com/widgets/widget_iframe.2f70fb173b9000da126c79afe2098f02.html?origin=https%3A%2F%2Fsec-deadlines.github.io', 'origin', 'https://sec-deadlines.github.io', -1) False
('https://syndication.twitter.com/settings?session_id=3dde761aca53da20aa18be88d09e4c4f7661dc50', 'session_id', '3dde761aca53da20aa18be88d09e4c4f7661dc50', -1) False
```

This output means the `origin` parameter in the request `https://platform.twitter.com/widgets/widget_iframe.2f70fb173b9000da126c79afe2098f02.html?origin=https%3A%2F%2Fsec-deadlines.github.i` is not a tracking request field by by Duumviri, similarily, the `session_id` paramter in `https://syndication.twitter.com/settings?session_id=3dde761aca53da20aa18be88d09e4c4f7661dc50` is not a mixed tracking request field. 

This example site took 24 mins, using a single 2.6 GHz core.

# Replicating Non-mixed Tracker Evaluation on EasyList and EasyPrivacy 
Input: none

Output: accuracy in label replication

Command: 
```bash
./replicate_filter_list_exp.sh
```

The script will print out the accuracy at the end. The number should be similar to the tentative accuracy reported of 96.53% in Section V.A.2.

This script took 11 mins on our machine.

# Inspecting Mixed Tracker Evaluation

Please refer to mixed_tracker_eval/README.md

# Citation
If you use the code/data in your research, please cite our work as follows:

```
@inproceedings{Siby22WebGraph,
  title     = {WebGraph: Capturing Advertising and Tracking Information Flows for Robust Blocking},
  author    = {Sandra Siby, Umar Iqbal, Steven Englehardt, Zubair Shafiq, Carmela Troncoso},
  booktitle = {USENIX Security Symposium (USENIX)},
  year      = {2022}
}
```

# Contact
For any inquiries, please contact He (Shawn) Shuang (8759shuang@gmail.com)



