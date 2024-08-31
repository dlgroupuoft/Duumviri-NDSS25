
from datetime import datetime
import subprocess
from glob import glob
import traceback
from difflib import SequenceMatcher
import requests
import mimeparse
import itertools
import utils
from tldextract import extract
import urllib
import os
from urllib.parse import urlparse
import selenium
from selenium.webdriver.common.by import By
from seleniumwire.utils import decode as wire_decode
import code 
import re
from adblockparser import AdblockRules, AdblockRule
import cssselect

class FilterListInterceptor():
	def __init__(self, filter, top_domain, for_wpcontent=False, fuzzy_matching_requests=False):
		self.filter = filter
		self.top_domain = top_domain
		self.for_wpcontent = for_wpcontent
		if isinstance(filter, str):
			self.filter = filter.split('\n')

		self.element_filter_regex = re.compile(r"""(?P<negation>~?)(?P<urls>[^#]*)(?P<type>##|#@#|#\$#)(?P<selector>.*)""")

		self.url_filters = AdblockRules(filter)
		self.ele_filters = self.load_ele_filters()
		self.disabled_ele_filters = []
		self.no_support_filters = self.check_no_support_filters()

		self.fuzzy_match_target = None
		if fuzzy_matching_requests:
			self.fuzzy_match_target = filter[0]
			
		if sum([len(self.url_filters.rules), len(self.ele_filters)]) == 0:
			print("Error loading rules: URL:%d ELE:%d no_support:%d from %d rules" % (len(self.url_filters.rules), len(self.ele_filters), len(self.no_support_filters), len(self.filter)))
		# if not ((len(self.url_filters.rules) + len(self.ele_filters) + len(self.no_support_filters)) == len(self.filter)):
		# 	print("Unloaded filters %s" % str([x for x in filter if (x not in [y.raw_rule_text for y in self.url_filters.rules]) and not self.is_element_filter(x)])) 
		self.clear()
	
	def flip_ele_rule(self, rule, exception_to_block):
		assert(isinstance(rule, tuple)) # an element filter

		after_pattern = '##'
		if not exception_to_block:
			after_pattern = '#@#'

		removed = False
		for index, r in enumerate(self.ele_filters):
			if r[-1] == rule[-1]: 
				tmp_rule = list(r)
				tmp_rule[2] = after_pattern
				self.ele_filters[index] =  tmp_rule
				removed = True

		if not removed:
			code.interact("Not flipped", local=dict(locals(), **globals()))
	
	def flip_exception_rule(self, rule, top_domain, exception_to_block=1):
		if isinstance(rule, AdblockRule):  # this is a URL filter
			# regular rule, domain is negation
			# 0, 0: very likely,
			# 0, 1: very unlikly for exceptional rule to have negated domains
			# 1, 0: equal likely, "||ad.doubleclick.net/ddm/clk/$domain=ad.doubleclick.net", this blocks RULE on that site. there isn't an usability issue (i.e. this can be a false data point)
			# 1, 1: equal likely, "||adskeeper.co.uk^$domain=~dashboard.adskeeper.co.uk", this blocks RULE on all other site, there isn't an usability issue we want to bloc 
			
			self.url_filters.flip_exception_rule(rule, top_domain, exception_to_block)
		else:
			self.flip_ele_rule(rule, exception_to_block)



	def clear(self):
		self.counter = [0, 0]
		self.blocked = [[], []]
		self.hit_rules = []
		self.wp_blocked = []
		# self.ele_filters = []
		self.passthough_counter = [0, 0]

	
	
	def check_no_support_filters(self):
		ret = []
		for block_item in self.filter:
			if '$popup' in block_item:
				ret.append(block_item)
		return ret
	
	def is_element_filter(self, rule_string):
		return ('##' in rule_string or '#@#' in rule_string or '#$#' in rule_string ) and not rule_string.strip().startswith('!')
	
	def similarity_score(self, t1, t2):
		return SequenceMatcher(None, t1, t2).quick_ratio()

	def interceptor(self, request):
		res = False
		if request.url.startswith('https://web.archive.org/web/'): 			
			orig_request_url = request.url
		
			new_url = request.url[28 + request.url[28:].find('/') + 1:]
			if new_url[:2] == '//':
				new_url = new_url.replace('//', 'https://')
			request.url = new_url

		print(request.url)
		if self.url_filters.should_block(request.url) or (self.fuzzy_match_target and self.similarity_score(request.url, self.fuzzy_match_target) >= 0.9):
			self.counter[0] += 1
			self.blocked[0].append(request)
			request.abort()
			res = True
		return res
		# print("Interceptor [%s] %s " % (str(res), request.url))
			
		
	def should_block_by_wp(self, url):
		return 'wp-content/themes' in url or 'wp-content/uploads' in url

	# def response_interceptor(self, request, response):
	# 	# https://web.archive.org/web/20220819211920/https://www.albumoftheyear.org/discover/
		
	# 	new_url = None

	# 	if request.url.startswith('https://web.archive.org/web/'): 			
	# 		orig_request_url = request.url
		
	# 		new_url = request.url[28 + request.url[28:].find('/') + 1:]
	# 		if new_url[:2] == '//':
	# 			new_url = new_url.replace('//', 'https://')

	# 		# if not request.url.startswith('http'):
	# 		# 	print('[%d] orig:%s\nafter:%s' % (response.status_code, orig_request_url, request.url))
	# 		# else:
	# 		# 	print(request.url)

	# 	# to_check_url = request.url if new_url is None else new_url
	# 	# print("\tTo_check_url: %s" % to_check_url)
	# 	# (self.approximate and self.)
	# 	if self.url_filters.should_block(to_check_url, options={'domain': self.top_domain}) or (self.for_wpcontent and self.should_block_by_wp(to_check_url)) or False:
	# 		self.counter[0] += 1
	# 		request.actual_response = response
	# 		self.blocked[0].append(request)
	# 		request.abort()

	# 	self.passthough_counter[1] += 1

	def interceptor_return_rule(self, request):
		mat_result = self.url_filters.should_block(request.url)
		if isinstance(mat_result, bool) and mat_result:
			# print("\tBlocked general re: %s " % request.url)
			self.counter[0] += 1
			request.abort()
		elif mat_result:
			# print("\tBlocked %s: %s" % (mat_result, request.url))
			self.counter[0] += 1
			request.abort()
			self.hit_rules.append(mat_result)

	def is_valid_css_selector(self, selector):
		try:
			cssselect.parse(selector)
		except cssselect.SelectorSyntaxError:
			print("\tNot a valid selector %s" % selector)
			return False
		return True

	def remove_element(self, driver, selector):
		if not self.is_valid_css_selector(selector):
			return 
		try:
			eles = driver.find_elements(by=By.CSS_SELECTOR, value=selector)
		except selenium.common.exceptions.InvalidSelectorException:
			print("\tInvalid selector %s" % selector)
			return 
			
		# if not eles:
		# 	print("\tNo elements found with selector %s " % selector)
		# 	return 
		self.counter[1] += len(eles)
		for ele in eles:
			self.blocked[1].append((ele.get_attribute('outerHTML'), ele.rect, ele.is_displayed(), ele.is_enabled()))

		driver.execute_script("""
		var elements = document.querySelectorAll(arguments[0]);
		for (let i =0; i < elements.length; i++) {
			elements[i].parentNode.removeChild(elements[i]);	
		}
		""", selector)
		# print("\tRemoving %d elements with %s" % (len(eles), selector))


	def domain_match(self, url, domain_csv):
		for target in domain_csv.split(','):
			if '*' in target:
				target_re = target.replace('*', '.*')
				target_re = target_re if target_re.startswith('.*') else '.*' + target_re
				target_re = target_re if target_re.endswith('.*') else  target_re + '.*'
				return re.match(target_re, url)
			else:
				u, t = extract(url), extract(target)
				if u.domain == t.domain and u.suffix == t.suffix:
					return True
		return False
	
	def load_ele_filters(self):
		ret = []
		for raw_rule_text in self.filter:
			if not self.is_element_filter(raw_rule_text):
				continue

			mat = re.match(self.element_filter_regex, raw_rule_text)
			if not mat:
				print("\tFailed to match element filter")
				# code.interact("Failed to match element filter", local=dict(locals(), **globals()))
				continue
			is_negation, domain_csv, rule_type, selector = len(mat.group('negation')) > 0, mat.group('urls'), mat.group('type'), mat.group('selector')
			assert(rule_type in ['##', '#@#', '#$#'])

			ret.append((is_negation, domain_csv, rule_type, selector, raw_rule_text))
		return ret
	
	def collect_hit_rules(self):
		return self.hit_rules
	

	def check_element_filters(self, driver, return_rule=False):
		for raw_rule_text in self.ele_filters:
			is_negation, domain_csv, rule_type, selector, orig_rule = raw_rule_text
			if rule_type in ['#$#', '#@#']: 
				continue
			
			if not domain_csv or domain_csv is None or self.domain_match(driver.current_url, domain_csv):
				if return_rule:
					if driver.find_elements(by=By.CSS_SELECTOR, value=selector):
						self.hit_rules.append(orig_rule)
				else:
					self.remove_element(driver, selector)
			
			# else:
			# 	if domain_csv:
			# 		print("\tDomain unmatch %s vs %s" % (driver.current_url, domain_csv))
				# code.interact(local=dict(locals(), **globals()))
		

class Cache():
	"""
	For masking parameters, use intercept 
	For rule-based blocking, use rule_interceptor
	For element-based blocking, use FilterListInterceptor 
	"""

	def __init__(self):
		self.proxy = []
		self.to_block = ""
		self.no_response = []
		self.not_in_proxy = []
		self.sessions = []
		self.IDX_BLOCKED = 0
		self.IDX_NOT_IN_PROXY = 1
		self.IDX_NO_RESPONSE = 2
		self.IDX_HIT = 3
		self.IDX_TIMES_COUNTER = 4
		self.IDX_BLOCKED_ELEMENTS = 5
		self.IDX_RESPONSES = 6
		self.IDX_TO_BLOCK = 7
		self.request_cache = {} # request -> targets in cache
		self.rule = None
	
	def get_last_blocked_requests(self):
		assert(self.sessions)
		return self.sessions[-1][self.IDX_BLOCKED]

	def get_last_blocked_eles(self):
		assert(self.sessions)
		return self.sessions[-1][self.IDX_BLOCKED_ELEMENTS]	

	def check_sessions(self):
		# code.interact('Invalid sessions', local=dict(locals(), **globals()))
		
		if len(self.sessions) > 4:
			self.sessions = self.sessions[:2] + self.sessions[-2:]
		
		assert(len(self.sessions) == 4)
		# if not len(self.sessions) == 4:
		# 	code.interact('Invalid sessions', local=dict(locals(), **globals()))

	def get_blocked_sessions(self):
		assert(self.sessions) 

		if len(self.sessions[0]) > 7: # IDX_TO_BLOCK is at index 7, means the length of th list is at least 8
			return list(filter(lambda x: x[self.IDX_TO_BLOCK][0] or x[self.IDX_TO_BLOCK][1], self.sessions))
		else:
			return list(filter(lambda x: x[self.IDX_BLOCKED] or (hasattr(self, 'IDX_BLOCKED_ELEMENTS') and x[self.IDX_BLOCKED_ELEMENTS]), self.sessions))
				
			

	def get_session_blocked_reqs(self):
		assert(self.sessions), 'No session get_session_blocked_reqs'
		blocked_sessions = self.get_blocked_sessions()
		assert(blocked_sessions), 'No blocked sessions'
		flattened = itertools.chain(*[session[self.IDX_BLOCKED] for session in blocked_sessions ])
		return utils.deduplicate_reqs(flattened)

	def get_session_blocked_eles(self):
		assert(self.sessions), 'No session get_session_blocked_eles'
		blocked_sessions = self.get_blocked_sessions()
		assert(blocked_sessions), 'No blocked sessions'
		if hasattr(self, 'IDX_BLOCKED_ELEMENTS'):
			flattened = itertools.chain(*[session[self.IDX_BLOCKED_ELEMENTS] for session in blocked_sessions ])
			return utils.deduplicate_eles(flattened)
		else:
			return []
		

	def is_valid_block(self, j):
		blocked_sessions = self.get_blocked_sessions()
		if j >= len(blocked_sessions):
			return False
		
		blocked_reqs = blocked_sessions[j][self.IDX_BLOCKED] 
		if hasattr(self, 'IDX_BLOCKED_ELEMENTS'):
			blocked_eles = blocked_sessions[j][self.IDX_BLOCKED_ELEMENTS]
		else:
			blocked_eles = []

		# if not ret:
		# 	code.interact("Invalid block", local=dict(locals(), **globals()))
		return blocked_reqs or blocked_eles




	def get_last_masked_requests(self):
		assert(self.sessions)
		return self.sessions[-1][self.IDX_RESPONSES]

	def set_block_rule(self, rule, top_domain):
		self.rule = FilterListInterceptor(rule, top_domain)
	
	def clear_block_rule(self):
		self.rule = None
		self.to_block = ''
		
	
	def flip_exception_rule(self, rule, top_domain, exception_to_block=1):
		self.rule.flip_exception_rule(rule, top_domain, exception_to_block)
	
	def set_block_url(self, url, top_domain_list):
		self.to_block = url
		self.to_domain_list = top_domain_list

		# if not self.filter_list:
		# 	self.filter_list = FilterListInterceptor(self.to_block, self.to_domain_list)
	
	
	def second_chance_matcher(self, request, parsed_r):
		# we cannot find the request in the proxy by parsing url
		# give it a second chance with more coarse searching 	
		for target in self.proxy:
			if parsed_r.path in target.url: 
				return target
		return None

	# def match_request_in_proxy(self)

	def similarity_score(self, r1, r2): # r1 is the target, r2 is the request 
		# matching the body
		score_body = SequenceMatcher(None, r1.body, r2.body).quick_ratio()
		scores = [1]

		# matching the headers
		for k, r1_v in r1.headers.items():
			if k in r2.headers:
				scores.append(SequenceMatcher(None, r1_v, r2.headers[k]).quick_ratio())
			else: 
				scores.append(0)
		tmp = [scores.append(0) for k, r2_v in r2.headers.items() if k not in r1.headers]

		penalty = 0
		if not r1.response or not r1.response.body: # discourage empty responses 
			penalty = -1
		return score_body + (sum(scores) / len(scores)) + penalty

	def url_match(self, u1, u2):
		return u1.strip("/") == u2.strip("/")
	
	def url_match_condition(self, target, request, thr):
		return SequenceMatcher(None, target.url, request.url).quick_ratio() > thr and target.method == request.method 

	
	def get_hash_key(self, req):
		return "%s %s" % (req.method, req.url)

	def check_request_from_cache(self, hash_key):
		if hash_key in self.request_cache:
			return self.request_cache[hash_key]
		return []

	def match_request_in_proxy(self, request, thr=0.90):
		hash_key = self.get_hash_key(request)
		targets = self.check_request_from_cache(hash_key)
		
		if not targets:
			for target in self.proxy:
				if 'parsed' not in dir(target):
					target.parsed = urlparse(target.url)
				
				if self.url_match_condition(target, request, thr):
					targets.append((self.similarity_score(target, request), target))


		# print("Request %s %s has %d targets" % (request.method, request.url, len(targets)))
		if not targets:
			return None
		else:
			self.request_cache[hash_key] = targets
		
		ret = None
		sort_by = 1
		if sort_by == 0: # sort by similarity 
			targets = sorted(targets, key=lambda x:x[0], reverse=True)
			ret = targets[0][1]
		elif sort_by == 1: # sort by time
			get_time_func = lambda x: x[1].date
			targets = sorted(targets, key=get_time_func, reverse=False) # all targets sorted by time from first to last 
			times = self.current_session_stats[self.IDX_TIMES_COUNTER][request.url]
			if times <= len(targets):
				ret = targets[times - 1][1]
			elif times > len(targets):
				ret = targets[-1][1]
			# print(times, ret)

		return ret

	def should_block_interceptor(self, request):
		assert((self.rule or self.to_block))
		
		if self.rule:
			return self.rule_interceptor(request)  # this is for element masking? 
		else: 
			# parameter masking					
			if (isinstance(self.to_block, list) or isinstance(self.to_block, tuple)) and (self.to_block[0] == '*' or request.url == self.to_block[0] or utils.fuzzy_match(request, self.to_block[0])) and request.url.startswith('http'):
				_, target_p_name, _, _ = self.to_block
				old_url, old_headers = request.url, dict(request.headers) 
				new_url, new_headers = utils.mask_param_in_req(request.url, request.headers, target_p_name)
				if new_url == old_url and new_headers == old_headers:
					# this happens when 
					return False

				# masked parameter is treated as blocked request for the prupose of feature extraction 
			
				# check if this is okay, otherwise have to make a seperate request and create response
				if request.method == 'GET':
					resp = requests.get(new_url, headers=new_headers, timeout=10) 
				elif request.method == 'POST':
					resp = requests.post(new_url, headers=new_headers, data=request.body, timeout=10)
				else:
					print('Cache unknown method %s' % request.method)

				self.current_session_stats[self.IDX_BLOCKED].append(request)
				self.current_session_stats[self.IDX_RESPONSES].append({'to_block': self.to_block, 'old_url':request.url, 'new_url':new_url, 'resp':resp})
				request.create_response(resp.status_code, list(resp.headers.items()), resp.content)
				# print('\tMasking [%s] into zero in %s' % (target_p_name, new_url))
				# print('\t\tStatus %d of size %d' % (resp.status_code, len(resp.content)))
				return True
			
			# whole req blocking
			if isinstance(self.to_block, str) and (request.url == self.to_block or utils.fuzzy_match(request, self.to_block)):
				self.current_session_stats[self.IDX_BLOCKED].append(request)
				request.abort()
				return True
			
		return False

	def interceptor(self, request):
		"""
		
		"""
		# update stats
		if request.url not in self.current_session_stats[self.IDX_TIMES_COUNTER]:
			self.current_session_stats[self.IDX_TIMES_COUNTER][request.url] = 0
		self.current_session_stats[self.IDX_TIMES_COUNTER][request.url] += 1
		
		try:
			if (self.to_block or self.rule) and self.should_block_interceptor(request):
				return 

			# response replay 
			replace_to = self.match_request_in_proxy(request)
			if replace_to is None:
				self.not_in_proxy.append(request)
				self.current_session_stats[self.IDX_NOT_IN_PROXY].append(request)
				request.abort()
				return 
			
			if replace_to.response is None: # blocking the request
				# self.blocked_requests.append(request)
				self.no_response.append(request)
				self.current_session_stats[self.IDX_NO_RESPONSE].append(request)
				request.abort()
				return 

			self.current_session_stats[self.IDX_HIT] += 1
			response = replace_to.response
			response_headers = response.headers if response is not None and response.headers is not None else {} 
			body = wire_decode(response.body, response.headers.get('Content-Encoding', 'identity'))			
			request.create_response(
				status_code=response.status_code,
				headers=response_headers.items(),
				body=body)
			
			# response refreshing 
			# if self.outf:
			# 	self.outf.write("Before %s\n" % request.response.headers.get_all("set-cookie"))
			# self.outf.write(str(type(request.response)))
			# self.outf.write(str(dir(request.response)))
			# request.response.refresh()
			# if self.outf:
			# 	self.outf.write("After %s" % request.response.headers.get_all("set-cookie"))

			assert(request.response)
		except:
			traceback.print_exc()

	def construction_request_response_interceptor(self, request, response):
		request.response = response
		self.proxy.append(request)
	
	def rule_interceptor(self, request):
		"""
		Work with self.rules
		"""
		res = False
		if self.rule.url_filters.should_block(request.url, {'domain': self.rule.top_domain}):
			self.current_session_stats[self.IDX_BLOCKED].append(request)
			request.abort()
			res = True
		# print('Rule: %s [%r] received: %s' % (self.rule.filter, res, request))
		return res
	
	def begin(self, driver):
		self.current_session_stats = [[], [], [], 0, {}, [], [], []]
		
		# which interceptor should be used? 	
		if not self.proxy:  			# recording mode, nothing to replay, always use this 
			if self.rule or self.to_block:
				driver.request_interceptor = self.should_block_interceptor

			driver.response_interceptor = self.construction_request_response_interceptor 
		else: # replay mode 
			# if not self.to_block and self.rule is None:
				# this can happen if during analysis, the block is done first and vanilla is done second 
				# code.interact("Attempting to block without first call update_to_block", local=dict(locals(), **globals()))

			# this interceptor is for replying and blocking 
			driver.request_interceptor = self.interceptor

		
	
	def print_stats(self):
		if len(self.sessions) > 1:
			print("Cache replay:%.2f BLOCKED:%d NOT_IN_PROXY:%d NO_RESPONSE:%d HIT:%d" % (self.sessions[-1][self.IDX_HIT] / max(1, len(self.sessions[-1][self.IDX_NOT_IN_PROXY]) + self.sessions[-1][self.IDX_HIT]), len(self.sessions[-1][self.IDX_BLOCKED]), len(self.sessions[-1][self.IDX_NOT_IN_PROXY]), len(self.sessions[-1][self.IDX_NO_RESPONSE]), self.sessions[-1][self.IDX_HIT]))
		else:
			print('Cache record: %d' % len(self.proxy))

	def end(self, driver):
		del driver.request_interceptor, driver.response_interceptor					

		# check ele filter 
		if self.rule and self.rule.ele_filters:
			self.rule.check_element_filters(driver)
			self.current_session_stats[self.IDX_BLOCKED_ELEMENTS] = self.rule.blocked[1]
		
		self.current_session_stats[self.IDX_TO_BLOCK] = (self.to_block, self.rule)
		# print("Cache end with toblock:%s" % str(self.to_block))

		self.sessions.append(self.current_session_stats)

		# self.print_stats()
	