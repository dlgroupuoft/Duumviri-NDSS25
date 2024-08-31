import code
import trio
import pickle
import compress_pickle
import code
import traceback
import pandas as pd
import networkx as nx
from collections import Counter
import seleniumwire
import cv2
import seleniumwire.undetected_chromedriver as wire_uc
import urllib
import numpy as np
from selenium.webdriver.chrome.options import Options as ChromeOptions
import os
import string
from selenium import webdriver
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from tldextract import extract
import re
import random
import time
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from difflib import SequenceMatcher
import requests
from adblockparser import AdblockRules, AdblockRule

parser = None
alexa_1m = None


class EasyPrivacyFilter():
    def __init__(self, filter_url='./blocking_filters/easyprivacy.txt'):
        if filter_url.startswith('http'):
            raw_easy_privacy = requests.get(filter_url).text.split('\n')
        elif os.path.isfile(filter_url):
            with open(filter_url) as inf:
                raw_easy_privacy = inf.readlines()
        else:
            assert (False), "Not compatible %s" % filter_url
        self.rules = AdblockRules(raw_easy_privacy)

    def should_block(self, url, options):
        return self.rules.should_block(url, options)


class ExternalFilters():
    def __init__(self, turnon=[1, 1, 1, 1, 1, 1]):
        self.turnon = turnon
        self.l0 = turnon[0] and EasyPrivacyFilter()
        self.l1 = turnon[1] and EasyPrivacyFilter(
            filter_url='./blocking_filters/easylist.txt')
        self.l2 = turnon[2] and EasyPrivacyFilter(
            filter_url='./blocking_filters/fanboy-cookiemonster.txt')
        self.l3 = turnon[3] and EasyPrivacyFilter(
            filter_url='./blocking_filters/fanboy_social.txt')
        self.l4 = turnon[4] and EasyPrivacyFilter(
            filter_url='./blocking_filters/fanboy-annoyance.txt')
        if turnon[5]:
            with open("./blocking_filters/peter_lowes.txt") as inf:
                self.l5 = turnon[5] and [line.strip().lower() for line in inf]

    def should_block_list(self, scripts):
        return [self.should_block(s) for s in scripts]

    def should_block(self, script, top_domain=''):
        if not top_domain:
            top_domain = '.'.join(extract(script)[1:])

        l0r, l1r, l2r, l3r, l4r = (self.turnon[0] and self.l0.should_block(script, {'domain': top_domain}),
                                   self.turnon[1] and self.l1.should_block(
                                       script, {'domain': top_domain}),
                                   self.turnon[2] and self.l2.should_block(
                                       script, {'domain': top_domain}),
                                   self.turnon[3] and self.l3.should_block(
            script, {'domain': top_domain}),
            self.turnon[4] and self.l4.should_block(script, {'domain': top_domain}))

        l5r = False
        domain = extract(script)[1]
        if self.turnon[5]:
            for x in self.l5:
                if domain.lower() in x:
                    l5r = True

        return [l0r, l1r, l2r, l3r, l4r, l5r]


def similarity_score(t1, t2):
    if t1 is None or t2 is None or not t1 or not t2:
        return 1
    return SequenceMatcher(None, t1, t2).quick_ratio()


def execute_script_through_cdp2(driver, script):

    ret = driver.execute_cdp_cmd('Runtime.evaluate', {
                                 'expression': script, 'returnByValue': True, 'userGesture': True})
    return ret['result']


async def execute_script_through_cdp(driver, script, context=None):
    async with driver.bidi_connection() as connection:
        cdp_session, devtools = connection.session, connection.devtools
        if context:
            result, exception = await cdp_session.execute(devtools.runtime.evaluate(expression=script, return_by_value=True, await_promise=False, generate_web_driver_value=True, user_gesture=True))
        else:
            result, exception = await cdp_session.execute(devtools.runtime.evaluate(expression=script, return_by_value=True, await_promise=False, generate_web_driver_value=True, user_gesture=True, context_id=context))
    return result, exception


def execute_script_with_catch2(driver, script):

    result = execute_script_through_cdp2(driver, script)

    try:
        if result['type'] == 'undefined':
            return None
        if 'webDriverValue' in result:
            return result['webDriverValue']['value']
        else:
            return result['value']
    except:

        text = str(result).lower()
        if 'error' in text or 'exception' in text:
            print(text)
            traceback.print_exc()
            print("Error executing %s" % script[:100])

    return None


def execute_script_with_catch(driver, script, context=None):
    result, exception = trio.run(
        execute_script_through_cdp, driver, script, context)
    if exception:
        print("Error %s from executing %s " %
              (script, exception.exception.description))
        return None

    return result.web_driver_value if result.value is None and result.web_driver_value is not None else result.value


def get_max_height(driver):
    max_height = 0

    try:
        max_height = execute_script_with_catch2(
            driver, "Math.max( document.body.scrollHeight, document.body.offsetHeight); ")

    except:
        traceback.print_exc()

    if not max_height or max_height is None:
        max_height = 0

    max_height = max_height + 150
    if max_height < 450:
        max_height = 1080

    return max_height


def try_set_window_size(driver, w, h):

    ret = driver.execute_cdp_cmd('Browser.getWindowForTarget', {})

    bounds = ret['bounds']
    bounds['width'], bounds['height'] = w, h
    bounds['windowState'] = 'normal'

    driver.execute_cdp_cmd('Browser.setWindowBounds', {
                           'windowId': ret['windowId'], 'bounds': bounds})

    times = 0
    while (w != execute_script_with_catch2(driver, "window.outerWidth;") or h != execute_script_with_catch2(driver, "window.outerHeight;")) and times < 3:
        times += 1
        time.sleep(0.5)

        driver.execute_cdp_cmd('Browser.setWindowBounds', {
                               'windowId': ret['windowId'], 'bounds': bounds})
    after_w = driver.get_window_size()['width']
    after_h = driver.get_window_size()['height']
    if after_w != w or after_h != h:
        print("\tFailed setting window size to (%d,%d) now:(%d,%d)" %
              (w, h, after_w, after_h))


def dump(data, path, compressed=False):
    if compressed:
        with open(path, 'wb') as f:
            compress_pickle.dump(data, f, compression='bz2',
                                 set_default_extension=False)
    else:
        pickle.dump(data, open(path, 'wb'))


def load(path, overwrite=False, compressed=None):
    if compressed is None:
        compressed = path.endswith('pickle')

    if compressed:
        try:
            with open(path, 'rb') as f:
                return compress_pickle.load(f, compression='bz2')
        except (OSError, EOFError, ModuleNotFoundError):

            with open(path, 'rb') as f:
                data = pickle.load(open(path, 'rb'))

            if overwrite:
                dump(data, path)
            return data
    else:
        try:
            with open(path, 'r') as inf:
                return inf.read()
        except:
            return load(path, overwrite=False, compressed=True)


def add_protocol(url):
    if not url.startswith('http'):
        return 'https://%s' % url
    return url


def init_chrome_with_cdp(binary_path="", driver_path="", wire_options={}, profile_num=0, extension_path="", enable_wire=1, no_proxy=False):

    binary_path = './chromium/ungoogled-chromium_111.0.5563.65-1.1.AppImage'
    driver_path = './chromium/chromedriver_v111'

    dc = DesiredCapabilities.CHROME
    dc['loggingPrefs'] = {'browser': 'ALL'}

    use_uc = False
    if use_uc:
        chrome_options = ChromeOptions()
    else:
        chrome_options = seleniumwire.undetected_chromedriver.ChromeOptions()

    chrome_options.add_argument(
        '--remote-debugging-port=%d' % (random.randint(49152, 65535)))

    user_profile = '/tmp/profile_%s' % (
        "".join(random.choices(string.ascii_letters, k=10)))
    chrome_options.add_argument('user-data-dir=%s' % user_profile)

    chrome_options.add_argument('--disable-infobars')
    chrome_options.add_argument('--disable-notifications')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disk-cache-size=1')
    chrome_options.add_argument('--media-cache-size=1=1')
    chrome_options.add_argument('--disk-cache-dir=/dev/null')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-web-security')
    chrome_options.add_argument('--disable-site-isolation')
    chrome_options.add_argument('--disable-application-cache')
    chrome_options.add_argument('--no-default-browser-check')
    chrome_options.add_argument('--test-type')
    chrome_options.add_argument("--disable-browser-side-navigation")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(
        '--disable-features=PreloadMediaEngagementData,MediaEngagementBypassAutoplayPolicies')

    chrome_options.add_argument("--headless=new")

    chrome_options.add_argument('--disable-useAutomationExtension')
    chrome_options.add_experimental_option(
        "excludeSwitches", ["enable-automation"])
    chrome_options.add_argument("disable-blink-features=AutomationControlled")

    if extension_path and extension_path.endswith('crx'):
        chrome_options.add_extension(extension_path)
    elif extension_path and os.path.isdir(extension_path):
        chrome_options.add_argument("--load-extension=%s" % extension_path)

    chrome_options.binary_location = binary_path
    wire_options = {'disable_encoding': True,
                    'ignore_http_methods': ['OPTIONS'], 'verify_ssl': False}

    if enable_wire and use_uc:
        driver = wire_uc.Chrome(
            options=chrome_options, seleniumwire_options=wire_options, desired_capabilities=dc)
    elif enable_wire and not use_uc:
        driver = seleniumwire.webdriver.Chrome(
            driver_path, options=chrome_options, seleniumwire_options=wire_options, desired_capabilities=dc)

    else:
        driver = webdriver.Chrome(
            driver_path, options=chrome_options, desired_capabilities=dc)

    return driver


class FeatureExtractor():
    def __init__(self, driver):
        self.driver = driver
        self.text_nodes_js = """
function countText(node){
	var counter = 0;
	if (node == null){
		return 0;
	}
	if(node.nodeType === 3){
		counter++;
	}
	else if(node.nodeType === 1) { // if it is an element node, 
	   var children = node.childNodes;    // examine the children
	   for(var i = children.length; i--; ) {
		  counter += countText(children[i]);
	   }
	}
	return counter;  
}
countText(document.body);
		"""
        self.color_js = """
		el = document.querySelectorAll('*');
		colors = [];
		for (let n of el) {
			if (n.textContent) {
				colors.push(window.getComputedStyle(n).getPropertyValue('color'));    
			}
		}
		[...new Set(colors)];
		"""

        self.max_doc_js = """
		const body = document.body;
		const html = document.documentElement;
		Math.max(body.scrollHeight, body.offsetHeight,
  			html.clientHeight, html.scrollHeight, html.offsetHeight);
		"""

        self.fonts_js = """
		  let fonts = [];

  for (let node of document.querySelectorAll('*')) {

	if (!node.style)
	 	continue;

	for (let pseudo of ['', ':before', ':after']) {

	  let fontFamily = getComputedStyle(node, pseudo).fontFamily;

	  fonts = fonts.concat(fontFamily);

	}

  }

  // Remove duplicate elements from fonts array
  // and remove the surrounding quotes around elements
  [...new Set(fonts)];
"""

        self.js_list_al_event_handlers = """
var allElements = Array.prototype.slice.call(document.querySelectorAll('*'));
allElements.push(document); // we also want document events
allElements.push(window); // we also want document events
var types = [];

for (let ev in window) {
	if (/^on/.test(ev)) types[types.length] = ev;
}

elements = [];
var node_string = '';
for (let i = 0; i < allElements.length; i++) {
	var currentElement = allElements[i];
	

	// this is normal events
	for (let j = 0; j < types.length; j++) {
		if (typeof currentElement[types[j]] === 'function') {
			node_string = currentElement.outerHTML;
			if (currentElement == document) {
				node_string = 'document';
			} else if (currentElement == window) {
				node_string = 'window';
			}

			elements.push({
				"node": node_string,
				"type": types[j],
				"func": currentElement[types[j]].toString(),
			});
		}
		else if (currentElement[types[j]]) {
			console.log(currentElement[types[j]]);
		}
	}

	if (typeof currentElement._getEventListeners === 'function') {
	  evts = currentElement._getEventListeners();
	  if (Object.keys(evts).length >0) {
		for (let evt of Object.keys(evts)) {
		  for (k=0; k < evts[evt].length; k++) {
			node_string = currentElement.outerHTML;
			if (currentElement == document) {
				node_string = 'document';
			} else if (currentElement == window) {
				node_string = 'window';
			}

			elements.push({
			  "node": node_string,
			  "type": evt,
			  "func": evts[evt][k].listener.toString(),
			});
		  }
		}
	  }
	}

	// this is jquery events
	if (window.jQuery) {
	currentElement_jq = window.jQuery(currentElement);
	// gets the events associated to a DOM element
	listeners = window.jQuery._data(currentElement_jq.get(0), "events") || {};
	events = Object.keys(listeners);
		events.forEach((type) => {
		(listeners[type] || []).forEach(getHandlers, type);
	});
	// eslint-disable-next-line
	function getHandlers(e) {
		const type = this.toString();
		// gets event-handlers by event-type or namespace
		node_string = currentElement.outerHTML;
		if (currentElement == document) {
			node_string = 'document';5
		} else if (currentElement == window) {
			node_string = 'window';
		}
		elements.push({
			"node": node_string,
			"type": type,
			"func": e.handler.toString(),
		});
	}
	}
}

	elements.sort(function (a, b) { return a.type.localeCompare(b.type); });

  """

    def get_features(self):
        driver = self.driver
        d1 = self.get_output_related_features(driver)
        d2 = self.get_input_related_features(driver)

        d4 = self.get_references(driver)

        d1.update(d2)

        d1.update(d4)

        return d1

    def get_listeners2(self, driver):
        ret = []
        try:
            listeners = execute_script_with_catch2(
                driver, self.js_list_al_event_handlers)

            for listener in listeners:
                try:

                    ret.append(list(listener.values()))
                except:
                    pass
        except:
            pass

        return ret

    def save_other_works_features(self, d1, d2, path):

        d = {}
        blocked_d = d1
        non_blocked_d = d2
        if d2['blocked_requests'] or d2['blocked_elements']:
            blocked_d = d2
            non_blocked_d = d1

        assert (blocked_d['blocked_elements'] or blocked_d['blocked_requests'])
        assert (not non_blocked_d['blocked_requests']
                and not non_blocked_d['blocked_elements'])

        for req in non_blocked_d['blocked_requests']:
            if 'actual_response' not in dir(req):
                print('no actual_response')
                req.actual_response = None

        bs_blocked = BeautifulSoup(blocked_d['html'], 'html.parser')
        bs_non_blocked = BeautifulSoup(non_blocked_d['html'], 'html.parser')
        domain_extracted = extract(blocked_d['url'])

        d['request_type'] = [self.get_req_content_type(
            req) for req in blocked_d['blocked_requests']]
        d['ad_keywords'] = self.get_ad_keywords(blocked_d['blocked_requests'])

        d['num_parameters'] = self.get_total_parameters(
            blocked_d['blocked_requests'])
        d['num_ad_dimensions'] = len(re.findall(self.ad_dimension_pattern, ' '.join(
            [req.url + ' ' + req.body.decode('utf-8') for req in blocked_d['blocked_requests']])))
        d['percent_domain'] = sum([1 for req in blocked_d['blocked_requests'] if extract(
            req.url).domain == domain_extracted.domain])
        d['percent_subdomain'] = sum([1 for req in blocked_d['blocked_requests'] if extract(
            req.url).domain == domain_extracted.domain and extract(req.url).subdomain])
        d['fp_domain_in_req'] = sum([(req.body.decode('utf-8') + req.url).count(
            domain_extracted.domain) for req in blocked_d['blocked_requests']])

        d['num_redirects'] = self.find_num_redirects(
            blocked_d['blocked_requests'])
        d['num_storage_in_requests'] = self.find_times_storage_in_requests(
            blocked_d, domain_extracted.domain)

        d['num_eval'] = sum([req.actual_response.body.count(
            b'eval') + req.actual_response.body.count(b'function') for req in blocked_d['blocked_requests']])

        d['fp_func_in_response'] = self.get_fp_func(
            blocked_d['blocked_requests'], domain_extracted.domain)
        d['blocked_response_size'] = sum(
            [len(req.actual_response.body) for req in blocked_d['blocked_requests']])
        d['window_navigator'] = sum([req.actual_response.body.count(
            b'window.navigator') for req in blocked_d['blocked_requests']])
        d['percent_subdoc_blocked'] = sum([req.actual_response.body.count(
            b'<body') for req in blocked_d['blocked_requests']]) / len(bs_non_blocked.find_all('body'))
        d['num_text_nodes_change'] = abs(
            non_blocked_d['num_text'] - blocked_d['num_text'])

        d['num_potential_listeners'] = sum([req.actual_response.body.count(
            b'addEventListener(') for req in blocked_d['blocked_requests']])
        d['change_network_size'] = sum(
            [len(req.body) for req in blocked_d['blocked_requests']])
        d['num_requests_blocked'] = len(blocked_d['blocked_requests'])
        d['perc_requests_blocked'] = (
            len(blocked_d['blocked_requests']) / len(non_blocked_d['requests']))
        d['num_ele_blocked'] = len(blocked_d['blocked_elements'])

        return d

    def get_delta_related_feature(self, driver):
        dict = {'blocked_requests': [], 'blocked_elements': []}
        if 'interceptor_class' in dir(driver) and driver.interceptor_class is not None:
            dict['blocked_requests'], dict['blocked_elements'] = driver.interceptor_class.blocked
            for req in dict['blocked_requests']:
                assert ('actual_response' in dir(req))
        return dict

    def add_script_source_cdp(self, driver, listeners):
        driver.execute_cdp_cmd('Debugger.enable', {})

        unique_scriptids = set([listner['scriptId'] for listner in listeners])
        id_to_source_mapping = {}
        for scriptId in unique_scriptids:
            try:
                id_to_source_mapping[scriptId] = driver.execute_cdp_cmd(
                    'Debugger.getScriptSource', {'scriptId': scriptId})['scriptSource']
            except:
                pass

        ret = []
        for listener in listeners:
            try:
                listener.update(
                    {'source': id_to_source_mapping[listener['scriptId']]})
            except:

                pass
            finally:
                ret.append(listener)

        return ret

    def add_nodes_cdp(self, driver, listeners):

        unique_nodeids = set([listener['backendNodeId']
                             for listener in listeners])
        id_to_node = {}
        for nodeid in unique_nodeids:
            try:
                id_to_node[nodeid] = driver.execute_cdp_cmd(
                    'DOM.getOuterHTML', {'backendNodeId': nodeid})['outerHTML']
            except:
                traceback.print_exc()
                pass

        ret = []
        for listener in listeners:
            if listener['backendNodeId'] not in id_to_node:
                listener['node'] = ''
            else:
                listener['node'] = id_to_node[listener['backendNodeId']]
            ret.append(listener)
        return ret

    def get_listeners_selenium_cdp(self, driver):
        document_node = driver.execute_cdp_cmd('DOM.getDocument', {})
        self.document_node_id = document_node["root"]["nodeId"]
        body_object = driver.execute_cdp_cmd('DOM.resolveNode', {
                                             'nodeId': document_node["root"]["nodeId"], 'object_group': 'foobar'})
        listeners = driver.execute_cdp_cmd('DOMDebugger.getEventListeners', {
                                           'objectId': body_object['object']['objectId'], 'depth': -1, 'pierce': False})['listeners']
        return listeners

    async def get_listeners_bidi(self, driver):
        async with driver.bidi_connection() as session:
            cdp_session, devtools = session.session, session.devtools

            document_node = await cdp_session.execute(devtools.dom.get_document())
            body_object = await cdp_session.execute(devtools.dom.resolve_node(node_id=devtools.dom.NodeId(document_node["root"]["nodeId"]), object_group='foobar'))

            listeners = await cdp_session.execute(devtools.dom_debugger.get_event_listeners(body_object.object_id, -1, True))

            ret = [listener for listener in listeners if listener.backend_node_id]
        return ret

    def convert_obj_dict(self, l):
        ret = []
        for ele in l:
            if isinstance(ele, dict):
                ret.append(ele)
            else:
                ret.append(ele.__dict__)
        return ret

    def get_listeners(self, driver):
        try:
            listeners = self.get_listeners_selenium_cdp(driver)

            listeners = self.convert_obj_dict(listeners)
            listeners = self.add_script_source_cdp(driver, listeners)
            listeners = self.add_nodes_cdp(driver, listeners)

        except:
            traceback.print_exc()

        return listeners

    def get_browser_log_entries(self, driver):
        """get log entreies from selenium and add to python logger before returning"""
        loglevels = {'NOTSET': 0, 'DEBUG': 10, 'INFO': 20,
                     'WARNING': 30, 'ERROR': 40, 'SEVERE': 40, 'CRITICAL': 50}

        slurped_logs = driver.get_log('browser')

        return slurped_logs

    def get_references(self, driver):
        whole_doc = driver.execute_cdp_cmd(
            'DOM.getDocument', {'depth': -1, 'pierce': True})
        try:
            whole_doc_html = driver.execute_cdp_cmd(
                'DOM.getOuterHTML', {'nodeId': whole_doc['root']['nodeId']})
        except:
            whole_doc_html = ""
            traceback.print_exc()

        ret = {'filter': driver.interceptor_class if 'interceptor_class' in dir(driver) and driver.interceptor_class else None, 'html': driver.page_source, 'current_url': driver.current_url, 'requests': driver.requests, 'log': self.get_browser_log_entries(
            driver), 'get_url': driver.get_url, 'script_mapping': driver.script_mapping, 'cdp_network': driver.cdp_network, 'whole_doc_html': whole_doc_html, 'whole_doc': whole_doc, 'load_time': driver.load_time}

        frame_tree = driver.execute_cdp_cmd('Page.getFrameTree', {})
        ret['frametree'] = frame_tree

        return ret

    def get_timeouts_intervals(self, driver):
        timers, intervals, logs = [], [], []
        try:
            timers, intervals, logs = execute_script_with_catch2(driver,
                                                                 "{activeTimers, activeIntervals, logs}")
        except Exception as E:
            print('get_timeouts_intervals', E)
        return timers, intervals

    def get_elements_by_tags(self, driver, tag_attributes):

        tag_attributes_str = "["
        for tag, attribute in tag_attributes:
            if attribute is None:
                attribute = 'null'
            else:
                attribute = "'%s'" % attribute
            tag_attributes_str += "['%s', %s], " % (tag, attribute)
        tag_attributes_str = tag_attributes_str[:-2]
        tag_attributes_str += "]"

        full_script = """
function get_elements(tag, attribute=null) {
	var eles = document.getElementsByTagName(tag);
	var ret = [];

	for (var i = 0; i < eles.length; i++) {
		ele = eles[i];
		if (tag == 'iframe' | tag == 'script') {
			try {
				ret.push([ele.outerHTML, ele.getBoundingClientRect().toJSON(), ele.contentDocument.documentElement.innerHTML]);
			} catch (e) {
				ret.push([ele.outerHTML, ele.getBoundingClientRect().toJSON(), '']);
			}
		} else if (attribute == null) {
			ret.push([ele.outerHTML, ele.getBoundingClientRect().toJSON()]);
		} else {
			ret.push([ele.getAttribute(attribute), ele.getBoundingClientRect().toJSON()]);
		}
	}
	return ret;
}

tag_attributes = %s;
var ret = [];
for (var i = 0; i < tag_attributes.length; i++) {
	ret.push(get_elements(tag_attributes[i][0], tag_attributes[i][1]));
}
ret;
""" % (tag_attributes_str)

        return execute_script_with_catch2(driver, full_script)

    def get_tag_listeners(self, driver, selector):
        document_node_id = driver.execute_cdp_cmd(
            'DOM.getDocument', {})["root"]["nodeId"]
        if selector == 'document':

            document_obj = driver.execute_cdp_cmd(
                'DOM.resolveNode', {'nodeId': document_node_id, 'object_group': 'foobar'})
            listeners = driver.execute_cdp_cmd('DOMDebugger.getEventListeners', {
                                               'objectId': document_obj['object']['objectId']})['listeners']
            return listeners

        eles = driver.execute_cdp_cmd('DOM.querySelectorAll', {
                                      'nodeId': document_node_id, 'selector': selector})['nodeIds']
        listeners = []

        for button_nodeid in eles:

            button_object_id = driver.execute_cdp_cmd(
                'DOM.resolveNode', {'nodeId': button_nodeid})['object']['objectId']

            button_listeners = driver.execute_cdp_cmd('DOMDebugger.getEventListeners', {
                                                      'objectId': button_object_id})['listeners']
            listeners += button_listeners

        listeners = self.add_nodes_cdp(driver, listeners)
        return listeners

    def get_input_related_features(self, driver):
        listeners_cdp = self.get_listeners(driver)
        document_listeners = self.get_tag_listeners(driver, 'document')

        listeners2 = self.get_listeners2(driver)

        timers, intervals = [], []

        cookies_cdp, local_storage2, session_storage2 = get_storage_values(
            driver)

        return {'listeners_cdp': listeners_cdp, 'listeners2': listeners2, 'cookies': cookies_cdp, 'local_storage': local_storage2, 'session_storage': session_storage2, 'timers': timers, 'intervals': intervals, 'document_listeners': document_listeners}

    def execute_scripts(self, driver, scripts):
        """
        scripts is a list of script
        this function puts together scripts into a big script that returns an array to save 
        """
        full_script = """
var scripts = %s;
var ret = []
for (var i = 0; i < scripts.length; i++) {
	ret.push(eval(scripts[i]));
}
ret;
""" % str(scripts)

        return execute_script_with_catch2(driver, full_script)

    def get_output_related_features(self, driver):

        tag_attributs = [('img', 'src'), ('video', 'src'), ('audio', None), ('iframe', 'src'),
                         ('a', 'href'), ('canvas', None), ('button', None), ('script', 'outerHTML')]
        images, videos, audio, iframes, links, canvases, buttons, scripts = self.get_elements_by_tags(
            driver, tag_attributs)

        num_text, fonts, highest_pixel, body_text, colors = self.execute_scripts(
            driver, [self.text_nodes_js, self.fonts_js, self.max_doc_js, 'document.body.innerText;', self.color_js])
        return {'num_text': num_text, 'images': images, 'videos': videos, 'iframes': iframes, 'links': links, 'fonts': fonts, 'highest_pixel': highest_pixel,
                'body_text': body_text, 'colors': colors, 'canvases': canvases, 'audio': audio, 'buttons': buttons, 'scripts': scripts}


def str_to_fname(s):
    s = ''.join(urlparse(s)[1:])
    return "".join(x for x in s if x.isalnum())


def ele_get_attr_with_catch(ele, attr):
    try:
        return ele.get_attribute(attr)
    except:
        traceback.print_exc()
        return ''


def get_screenshot_as_numpy(driver):
    png = driver.get_screenshot_as_png()
    nparr = np.frombuffer(png, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def param_is_id(s, thr=1000000000):
    has_digit = any([c.isdigit() for c in s])
    has_lower = any([c.islower() for c in s if c not in ['x']])
    has_upper = any([c.isupper() for c in s])

    num_possibilities = 0
    if has_digit:
        num_possibilities += 10
    if has_lower:
        num_possibilities += 26
    if has_upper:
        num_possibilities += 26

    total_possibilities = num_possibilities ** len(s)

    return total_possibilities > thr


def param_from_storage(s, storage_values):
    return len(s) >= 5 and (s in storage_values or any([s in v for v in storage_values]))


def get_storage(driver, script):
    try:
        ret = driver.execute_script(script)
    except:
        return {}
    ret2 = {}
    for k, v in ret.items():
        if v:
            ret2[k] = v
    return ret2


def get_storage_values(driver):
    cookies_cdp, local_storage2, session_storage2 = [], {}, {}

    try:
        cookies_cdp = driver.execute_cdp_cmd(
            'Storage.getCookies', {})['cookies']
        local_storage2 = get_storage(driver, 'return window.localStorage;')

        session_storage2 = get_storage(driver, 'return window.sessionStorage;')
    except:
        traceback.print_exc()
    return cookies_cdp, local_storage2, session_storage2


def mask_param_in_req(request_url, request_headers, target_p_name):
    new_url, new_headers = request_url, dict(request_headers)

    if (isinstance(target_p_name, str) and target_p_name in request_url) or isinstance(target_p_name, bool):
        parsed = urlparse(request_url)
        parsed_params = urllib.parse.parse_qs(parsed.query)
        if isinstance(target_p_name, bool) and target_p_name:
            parsed_params = {}
        else:
            for p_name, p_value in parsed_params.items():
                if p_name == target_p_name:
                    parsed_params[p_name] = '0'
        new_query = urllib.parse.urlencode(parsed_params)
        new_url = urllib.parse.urlunparse(
            urlparse(request_url)._replace(query=new_query))
        if isinstance(target_p_name, bool):
            print('new url %s from old %s' % (new_url, request_url))

    elif isinstance(target_p_name, str) and target_p_name in request_headers.get('cookie', ''):
        cookie_header = request_headers.get('cookie', '')
        cookies_parsed, splitter = parse_raw_cookie(
            cookie_header, return_splitter=True)
        cookies_parsed[target_p_name] = '0'

        splitter += ' '
        new_headers['cookie'] = splitter.join(
            ['%s=%s' % (k, v) for k, v in cookies_parsed.items()])
    return new_url, new_headers

def get_total_parameters(reqs_urls):
    total = 0
    for req in reqs_urls:
        if isinstance(req, str):
            total += len(parse_qs(urlparse(req).query))
        else:
            total += len(parse_qs(urlparse(req.url).query))
    return total


def extract_graph_features(scripts, g, pd):
    """
    Script edge is list of script edges 
    each script edge is a list of tuples (u, v, k, data)
    """
    ret = []

    for script_edges in scripts:
        graph_num_requests_sent = 0
        graph_total_parameters_sent = 0
        graph_num_requests_complete = 0
        graph_num_event_listeners_installed = 0
        cookie_read_times, cookie_set_times = 0, 0
        storage_read_times, storage_set_times = 0, 0
        listeners, creates, attributes, bindings = Counter(), Counter(), Counter(), Counter()
        read_storage, write_storage = [], []
        cookie_delete_times, storage_delete_times = 0, 0
        script_url = ''
        page_url = ''

        for u, v, k, data, edge_type in script_edges:
            try:

                v_data, u_data = {}, {}
                if v:
                    v_data = g.nodes[v]
                    u_data = g.nodes[u]

                if k == 'script_url':
                    script_url = data['script_url']
                    page_url = data['page_url']

                if data['edge type'] == 'binding event':
                    for u1, v1, k1, data1 in g.out_edges(v, data=True, keys=True):
                        if data1['edge type'] != 'binding':
                            continue

                        binding_node = g.nodes[v1]
                        assert (binding_node['node type'] == 'binding')
                        bindings[binding_node['binding']] += 1
                if data['edge type'] == 'request start' and edge_type == 'out':
                    graph_num_requests_sent += 1
                    graph_total_parameters_sent += get_total_parameters(
                        v_data['url'])
                if data['edge type'] == 'request complete':
                    graph_num_requests_complete += 1

                if data['edge type'] == 'add event listener' and edge_type == 'out':
                    graph_num_event_listeners_installed += 1
                    if 'key' in data:
                        listeners[v_data['tag name'] + '.' + data['key']] += 1
                    else:
                        listeners[v_data['tag name'] + '.unknown'] += 1

                if data['edge type'] == 'storage read result':
                    storage_value = data['value'] if 'value' in data else ''
                    storage_key = data['key'] if 'key' in data else ''
                    if 'key' not in data:
                        print('%s no key' % pd.save_path)

                    read_storage.append(
                        (storage_key, storage_value, script_url, page_url))

                elif data['edge type'] == 'storage set':
                    storage_value = data['value'] if 'value' in data else ''
                    storage_key = data['key'] if 'key' in data else ''
                    if 'key' not in data:
                        print('%s no key' % pd.save_path)

                    write_storage.append(
                        (storage_key, storage_value, script_url, page_url))
                elif data['edge type'] == 'delete storage':
                    if v_data['node type'] == 'cookie jar':
                        cookie_delete_times += 1
                    elif v_data['node type'] == 'local storage':
                        storage_delete_times += 1

                if 'storage' in data['edge type'] and edge_type == 'out' and v_data['node type'] == 'cookie jar':
                    if data['edge type'] == 'storage set':
                        cookie_set_times += 1
                    else:
                        cookie_read_times += 1

                if 'storage' in data['edge type'] and edge_type == 'out' and v_data['node type'] == 'local storage':
                    if data['edge type'] == 'storage set':
                        storage_set_times += 1
                    else:
                        storage_read_times += 1

                if data['edge type'] in ['create node', 'insert node']:
                    if 'tag name' in v_data:
                        creates[v_data['tag name'] + '.create'] += 1
                    elif 'node type' in v_data:
                        creates[v_data['node type'] + '.create'] += 1
                    else:
                        creates['unknown.create'] += 1
                if data['edge type'] == 'set attribute':
                    attributes[v_data['tag name'] + '.' + data['key']] += 1

                if data['edge type'] == 'js call':
                    attributes[v_data['method']] += 1
            except:
                traceback.print_exc()
                continue

        stats = {}
        for var in ['graph_num_requests_sent', 'graph_total_parameters_sent',   'graph_num_event_listeners_installed', 'cookie_read_times', 'cookie_set_times',
                    'storage_read_times', 'storage_set_times', 'listeners', 'creates', 'attributes', 'bindings', 'script_url', 'page_url', 'read_storage', 'write_storage', 'cookie_delete_times', 'storage_delete_times']:
            stats[var] = locals()[var]

        ret.append((stats, script_edges))

    return ret


def extract_domain(url):
    return '.'.join(extract(url)[1:])


def get_top_domain_list(pd):
    fp_domain = extract_domain(pd.url)
    fp_domain2 = extract_domain(pd.vanilla_retry[0].rendered['current_url'])
    return list(set([fp_domain, fp_domain2]))


def pd_get_pagegraphs(pd):
    if not isinstance(pd, dict):
        graphs = pd.pagegraph
    else:
        graphs = pd
    graph_items = list(graphs.items())
    graph_items = sorted(graph_items, key=lambda x: len(x[1]), reverse=True)
    return graph_items


def load_g_from_pd(pd):
    pg = pd_get_pagegraphs(pd)

    if not pg:
        return
    try:

        gs = []
        for fname, graph_string_format in pg:
            try:
                gs.append((fname, nx.parse_graphml(graph_string_format)))
            except:

                pass
        if not gs:
            return

        g = gs[0][1]
        for fname, g1 in gs[1:]:
            try:
                if isinstance(g1, nx.classes.digraph.DiGraph):
                    g1 = nx.MultiDiGraph(g1)
                assert (isinstance(g1, nx.classes.multidigraph.MultiDiGraph))
                g = nx.disjoint_union(g, g1)
            except:
                traceback.print_exc()
                print('%s is not valid: %s' % (fname, str(type(g1))))
    except:
        traceback.print_exc()
        print('failed to load pagegraph %s' % pd.save_path)
        code.interact('load_g_from_pd', local=dict(locals(), **globals()))
        return None
    return g


def get_script_nodes_from_pg(g, hit_condition_func, pd=None, backwards=True, return_by_tuple=False):
    """
    If backwords is ture, we find initiating scripts
            - also deals with HTML elements and ifranes 

    if backwords is false, we find what the node does 
            - mostly just the script node itself  
    """
    script_nodes = set()
    for n in g.nodes:
        if g.nodes[n]['node type'] == 'resource' and hit_condition_func(g.nodes[n]['url']):
            if backwards:
                edges_to_check = g.in_edges(n, data=True, keys=True)
            else:
                edges_to_check = g.out_edges(n, data=True, keys=True)

            for node1, node2, k, data in edges_to_check:
                node_to_check = node1
                if node1 == n:
                    node_to_check = node2

                tracking_node_type = g.nodes[node_to_check]['node type']
                if tracking_node_type == 'script':

                    if return_by_tuple:
                        script_nodes.add((node_to_check, g.nodes[n]['url']))
                    else:
                        script_nodes.add(node_to_check)

                elif tracking_node_type == 'HTML element':
                    tracking_tag = g.nodes[node_to_check]['tag name']
                    if tracking_tag in ['img', 'link', 'script']:
                        predecessors = list(nx.bfs_edges(
                            g.reverse(), source=node_to_check, depth_limit=1))
                        for _, predee in predecessors:
                            if g.nodes[predee]['node type'] == 'parser':
                                continue
                            if g.nodes[predee]['node type'] == 'script':

                                if return_by_tuple:
                                    script_nodes.add(
                                        (predee, g.nodes[n]['url']))
                                else:
                                    script_nodes.add(predee)
                    else:
                        if pd:
                            print('[%s] unhandled tag name %s' % (
                                pd.save_path, g.nodes[node_to_check]['tag name']))
                elif tracking_node_type in ['DOM root', 'parser']:
                    predecessors = list(nx.bfs_edges(
                        g.reverse(), source=node_to_check, depth_limit=2))
                    for _, predee in predecessors:
                        if g.nodes[predee]['node type'] == 'parser':
                            continue
                        if g.nodes[predee]['node type'] == 'script':

                            if return_by_tuple:
                                script_nodes.add((predee, g.nodes[n]['url']))
                            else:
                                script_nodes.add(predee)
                else:
                    if pd:
                        print('[%s] unhandled node init request %s' %
                              (pd.save_path, tracking_node_type))

    return script_nodes


def get_script_edges(g, script_nodes, pd=None):
    scripts = []
    for script_node in set(script_nodes):
        if 'url' in g.nodes[script_node]:
            script_url = g.nodes[script_node]['url']
        else:
            script_url = 'inline'

        if pd:
            one_script = [('', '', 'script_url', {'edge type': 'debug', 'script_url': script_url,
                           'page_url': pd.url, 'script_node': g.nodes[script_node], 'path': pd.save_path}, 'debug')]
        else:
            one_script = []

        out_edges = g.out_edges(script_node, data=True, keys=True)
        for u, v, k, data in out_edges:

            one_script.append((u, v, k, data, 'out'))

        in_edges = g.in_edges(script_node, data=True, keys=True)
        for u, v, k, data in in_edges:

            one_script.append((u, v, k, data, 'in'))

        scripts.append(one_script)
    return scripts


def fuzzy_match(req, target):
    """
    req is a request object 
    target is a string of the url to block
    """
    if isinstance(req, str):
        req_url = req
    else:
        req_url = req.url
    p1 = urlparse(req_url)
    p2 = urlparse(target)

    u1 = p1._replace(query="").geturl()
    u2 = p2._replace(query="").geturl()
    url_simi = similarity_score(u1, u2) > 0.9 and os.path.basename(
        p1.path) == os.path.basename(p2.path)

    param_simi = True
    if p1.query or p2.query:
        param_simi = set(urllib.parse.parse_qs(p1.query).keys()) == set(
            urllib.parse.parse_qs(p2.query).keys())

    return url_simi and param_simi


def parse_raw_cookie(cookie, return_splitter=False):
    """
    cookie format is x=1&y=2&z=3
    return a dict of {x:1, y:2, z:3}
    """
    if not cookie:
        return {}

    splitter = ''
    if '&' in cookie:
        splitter = '&'
    if ':' in cookie:
        splitter = ':'
    if ';' in cookie:
        splitter = ';'
    splitted = [cookie]
    if splitter:
        splitted = cookie.split(splitter)

    def split_by_equal_sign(s):
        if '=' not in s:
            return {'': s}
        k, v = s.split('=', 1)
        return {k.strip(): v.strip()}

    lst_of_singles = list(map(split_by_equal_sign, splitted))

    merged = {k: v for d in lst_of_singles for k, v in d.items()}

    if return_splitter:
        return merged, splitter
    return merged


def get_cookie_values(raw_cookies):
    """
    takes a list of cookies 
    return the cookie valus
    """
    ret = []
    for cookie in raw_cookies:
        cookie_values = list(parse_raw_cookie(cookie['value']).values())
        ret += cookie_values
    return ret


def deduplicate_reqs(reqs):

    d = {req.url: req for req in reqs}
    return list(d.values())


def deduplicate_eles(eles):
    return eles


def merge_feature_stats(list_of_feature_stats):
    ret = Counter()
    if not list_of_feature_stats:
        return ret
    f1 = list_of_feature_stats[0]
    for k in f1.keys():
        if k in ['script_url', 'page_url']:
            continue

        for f in list_of_feature_stats:
            assert (k in f)
            if k in ret:
                ret[k] += f[k]
            else:
                ret[k] = f[k]
    return ret


def find_all_pagedeltas(path, existing=None, first_k=-1):
    """
    existing is a list of csv fnames in tmp_file_path 
    """

    pagedelta_fnames = []
    skipped = []
    total = 0
    for dpath, dirs, files in os.walk(path):
        pagedelta_fname = ''
        for f in files:
            if f.endswith('pagedelta.pickle'):
                sitename = os.path.basename(dpath)
                pagedelta_fname = os.path.join(dpath, f)
                total += 1
                if existing is None or sitename not in existing:
                    if first_k != -1:
                        site_idx = int(sitename.split('_')[0])
                        lower = 1000 * (first_k - 1)
                        upper = 1000 * first_k
                        if site_idx >= lower and site_idx < upper:
                            pagedelta_fnames.append(pagedelta_fname)
                    else:
                        pagedelta_fnames.append(pagedelta_fname)
                else:
                    skipped.append(sitename)

    return pagedelta_fnames


class FeatureCollector():
    pass


def calculate_score(row, numeric_columns):

    row_nums = row[numeric_columns]
    return row_nums.abs().sum()


def aggregate_func(column):
    if column.dtype in [str, bool, object]:
        try:
            ret = column.iloc[0]
        except:
            traceback.print_exc()
            code.interact("aggregate_func", local=dict(locals(), **globals()))
        return ret
    elif column.dtype in [np.int64, np.float64]:
        return min(abs(column))
    else:
        print(column)


def check_reset(column, debug_urls=None):

    if column.name in ['num_requests_blocked',	'percent_requests_blocked',	'total_parameters',	'num_ad_dimensions', 'times_fp_in_blocked_requests',	'num_fp_req_blocked',	'num_tp_req_blocked',	'num_ad_keywords',	'num_storage_values_out',	'total_response_size',	'avg_response_size',	'sensitive_fp_v', 'sensitive_tp_v', 'debug_name',	'debug_blocked_url',	'debug_is_ad', 'debug_is_tracker', 'debug_to_block', 'debug_url', 'index']:
        return column

    check_by_url = debug_urls is not None
    if check_by_url:
        df = pd.merge(column, debug_urls, right_index=True, left_index=True)
        try:
            unique_debug_urls = df.groupby('debug_url').filter(lambda x: len(
                x[column.name].unique()) == 1).loc[:, 'debug_url'].unique()
        except:
            traceback.print_exc()

        if len(unique_debug_urls):
            df.loc[df['debug_url'].isin(unique_debug_urls), [column.name]] = 0
            return df[column.name]

    else:
        uniuqes = column.unique()

        if len(uniuqes) == 1 and uniuqes[0] != 0:

            return [0] * len(column)
    return column


def reset_identical_values(df):

    assert ('debug_url' in df.columns)
    ret = df.apply(check_reset, axis=0, debug_urls=df.loc[:, 'debug_url'])

    return ret


def combine_retries(df):
    merge_method = 0
    merge_method = 1
    merge_method = 2

    if merge_method == 0:

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = numeric_columns.drop(['debug_vanilla_retry', 'debug_blocked_retry', 'total_blocked_size',
                                               'percent_requests_blocked', 'total_response_size', 'total_requests', 'avg_response_size', 'height_diff'], errors='ignore')
        numeric_columns = list(numeric_columns[:40])

        df['score'] = df.apply(
            calculate_score, args=(numeric_columns,), axis=1)
        ret = df.sort_values(['debug_name', 'score']
                             ).groupby('debug_name').head(1)

    elif merge_method == 1:

        ret = df.groupby('debug_name').agg(aggregate_func).reset_index()
        assert (ret.shape[1] == df.shape[1])
    elif merge_method == 2:
        ret = df.groupby('debug_name').head(1)

    should_reset = False
    if should_reset:
        ret = reset_identical_values(ret)
    return ret


def equal_sized_chunks(l, k):
    lists = []
    for i in range(k):
        lists.append([])

    rows_per_df = len(l) // k
    lists = []
    for i in range(k):
        start_row = i * rows_per_df
        end_row = (i + 1) * rows_per_df if i < k - 1 else len(l)
        val = l.iloc[start_row:end_row, :] if isinstance(
            l, pd.DataFrame) else l[start_row: end_row]
        lists.append(val)
    return lists
