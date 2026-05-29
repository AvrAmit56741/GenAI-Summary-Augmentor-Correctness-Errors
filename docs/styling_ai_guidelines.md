# Code Review Guidelines

## Table of Contents
1. [Reduce Coupling in function parameters](#001-reduce-coupling-in-function-parameters)
2. [Look around for standards](#002-look-around-for-standards)
3. [Don't bypass bugs](#003-dont-bypass-bugs)
4. [Each class in its own module](#004-each-class-in-its-own-module)
5. [Clear TODOs](#005-clear-todos)
6. [Handle all branches explicitly](#006-handle-all-branches-explicitly)
7. [Each parameter in its own line](#007-each-parameter-in-its-own-line)
8. [Keep your private stuff private](#008-keep-your-private-stuff-private)
9. [Reduce coupling in object fields](#009-reduce-coupling-in-object-fields)
10. [Use OOD](#010-use-ood)
11. [Don't hide errors](#011-dont-hide-errors)
12. [No hard coded configuration](#012-no-hard-coded-configuration)
13. [Limit indentation](#013-limit-indentation)
14. [Provide error details](#014-provide-error-details)
15. [Boolean condition](#015-boolean-condition)
16. [Initialize nested collections](#016-initialize-nested-collections)
17. [Use type hints - Avoid generic types](#017-use-type-hints---avoid-generic-types)
18. [Dataclass over complex dicts](#018-dataclass-over-complex-dicts)
19. [Informative naming](#019-informative-naming)
20. [Extract config once at method start](#020-extract-config-once-at-method-start)
21. [Let helper methods extract their own data](#021-let-helper-methods-extract-their-own-data)
22. [Follow the established pattern](#022-follow-the-established-pattern)
23. [Avoid unnecessary empty checks before loops](#023-avoid-unnecessary-empty-checks-before-loops)
24. [Use datetime instead of epoch timestamps](#024-use-datetime-instead-of-epoch-timestamps)
25. [Return values and error handling](#025-return-values-and-error-handling)
26. [Break from function ASAP](#026-break-from-function-asap)
27. [Function length and extraction](#027-function-length-and-extraction)
28. [Logging](#028-logging)
29. [Prefer simple solutions](#029-prefer-simple-solutions)
30. [Avoid code duplication (DRY)](#030-avoid-code-duplication-dry)
31. [Never use mutable default arguments](#031-never-use-mutable-default-arguments)
32. [Use context managers for resource management](#032-use-context-managers-for-resource-management)
33. [Catch specific exceptions](#033-catch-specific-exceptions)
34. [Use Enum instead of string constants](#034-use-enum-instead-of-string-constants)
35. [Dependency injection](#035-dependency-injection)
36. [All imports at the top of the file](#036-all-imports-at-the-top-of-the-file)

---

## #001 Reduce Coupling in function parameters

Design functions to receive as few parameters as possible. This reduces the coupling between the calling code and the function.

**Example:**

The `agent_id` is already part of `agent_item`, so avoid passing it as a separate parameter.

```python
# ❌ BAD - agent_id is redundant, it's already inside agent_item
agent_id = resolve_agent_identifier()
agent_item = HomeGrownAgent(
    creation_time=now,
    agent_id=agent_id,
    agent_name=input_event.agent_name,
)
add_tools(agent_item, agent_id)  # ❌ agent_id is already in agent_item

# ✅ GOOD - extract agent_id inside the function
def add_tools(agent_item: AgentItem) -> None:
    agent_id = agent_item.agent_id
    ...

add_tools(agent_item)
```

---

## #002 Look around for standards

Always look around for standards before making a change. Someone invested thought in this code and decided on a standard — continue using it rather than introducing a new one.

**Python naming conventions (PEP 8):**
- `snake_case` for functions, methods, variables, module names
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- `_leading_underscore` for private members

**IMPORTANT! Always use `snake_case`, even for acronyms:**
- URL should be `url`, not `URL` (in variable/function names)
- ID should be `id`, not `ID`
- HTTP should be `http`, not `HTTP`

**BUT do not use snake_case for user-facing messages.** User-facing strings should use proper capitalization and spacing.

**Example:**

If all Kafka topic constants follow the pattern `KAFKA_TOPIC_*_AGENTS`, do not break this pattern:

```python
KAFKA_TOPIC_EVENTS_ENDPOINTS = "events-endpoints"
KAFKA_TOPIC_EVENTS_AGENTS = "events-agents"
KAFKA_TOPIC_INVENTORY_AGENTS = "inventory-agents"
KAFKA_TOPIC_AGENTIC_TOOL_CALL = "agentic-tool-calls"  # ❌ Breaks naming convention
KAFKA_TOPIC_FLOW_RISKS = "flows-agents"               # ❌ Breaks naming convention
```

---

## #003 Don't bypass bugs

If we have bugs, we want to raise errors, not avoid them. Do not create code that silently ignores bugs.

**Example:**

```python
# ❌ BAD - bypassing a bug instead of reporting it
def upsert_tools_in_po(self) -> None:
    if self.persist_api is None:  # Hides that the caller forgot to set persist_api
        return

# ✅ GOOD - raise so the bug is visible
def upsert_tools_in_po(self) -> None:
    if self.persist_api is None:
        raise ValueError("persist_api must be initialized before calling upsert_tools_in_po")
```

---

## #004 Each class in its own module

Each class should live in its own module (file). Do not place more than a single class in a module.

**Example:**

```
❌ BAD
protection.py
    class AgenticProtectionBehavior: ...
    class AgenticProtectionGuards: ...   ← two classes in one file

✅ GOOD
agentic_protection_behavior.py
    class AgenticProtectionBehavior: ...

agentic_protection_guards.py
    class AgenticProtectionGuards: ...
```

---

## #005 Clear TODOs

Do not leave TODOs in the code.

**Example:**

```python
# ❌ BAD
def describe_tool_naturally(self, tool_name: str, tool_arguments: str) -> str:
    user_message = f"Tool: {tool_name}\nArguments: {tool_arguments}"
    # TODO: ADD CACHING  ❌ Remove TODOs before merging
    return self.provider.send_chat_completion(TOOL_DESCRIPTION_SYSTEM_PROMPT, user_message)
```

---

## #006 Handle all branches explicitly

Do not assume the default branch in an `if/elif` chain or `match` statement is something that should not be handled. If reaching the default is a programming error, raise an exception.

**Example:**

```python
# ❌ BAD - silently returns the input, masking an invalid value
def get_owasp_category_for_guard(guard_type: str) -> str:
    match guard_type:
        case "pii":
            return "identity"
        case "topics":
            return "agent_behaviour_hijack"
        case "injection":
            return "identity"
        case _:
            return guard_type  # ❌ Should raise an error

# ✅ GOOD
def get_owasp_category_for_guard(guard_type: str) -> str:
    match guard_type:
        case "pii":
            return "identity"
        case "topics":
            return "agent_behaviour_hijack"
        case "injection":
            return "identity"
        case _:
            raise ValueError(f"Unknown guard type: {guard_type!r}")
```

---

## #007 Each parameter in its own line

Each function parameter should be on its own line, even when there is only one parameter.

**Example:**

```python
# ❌ BAD
def get_owasp_category_for_guard(guard_type: str) -> str:

# ✅ GOOD
def get_owasp_category_for_guard(
    guard_type: str,
) -> str:

# Multiple parameters
def create_agent(
    agent_id: str,
    agent_name: str,
    model_name: str,
) -> Agent:
```

---

## #008 Keep your private stuff private

Use public access only when you must expose something. Prefix private methods, functions, and attributes with a single underscore (`_`).

**Example:**

```python
# ❌ BAD
class Analyzer:
    def AddToolCountCalls(self, input_event: AgenticEventInput) -> None:  # ❌ Public + wrong naming
        ...

    IsEnabledLogs = True  # ❌ Should be private

# ✅ GOOD
class Analyzer:
    _is_enabled_logs: bool = True

    def _add_tool_count_calls(self, input_event: AgenticEventInput) -> None:
        ...
```

---

## #009 Reduce coupling in object fields

The purpose of an object is that fields are shared across all its methods. Do not pass fields as parameters to methods that already have access to them via `self`.

**Example:**

```python
# ❌ BAD - self.po_with_id is already a field
def run_risks_if_required(self) -> None:
    now = self.now_time.now()
    if not self._should_run_risks_by_po(now, self.po_with_id):  # ❌ redundant param
        return

# ✅ GOOD
def run_risks_if_required(self) -> None:
    now = self.now_time.now()
    if not self._should_run_risks_by_po(now):
        return

def _should_run_risks_by_po(self, now: datetime) -> bool:
    # access self.po_with_id directly
    ...
```

---

## #010 Use OOD

When creating a new component that takes common input parameters used across its methods, do not pass those parameters repeatedly. Create a class, store the inputs as fields, and let methods access them via `self`.

**Example:**

#### ❌ BEFORE: Function-Based Approach (Repetitive Parameters)

```python
# Every function needs the same parameters passed over and over
def analyze_csp(header_value: str, request_origin: str) -> None:
    parsed_policy = parse_content_security_policy(header_value)

    # Pass parsed_policy to every function - REPETITIVE
    xss_result = classify_xss_mitigation(parsed_policy)
    tls_result = classify_tls_enforcement(parsed_policy)

    # These need BOTH - even more repetitive
    frame_result = classify_frame_ancestors(parsed_policy, request_origin)
    form_result = classify_form_action(parsed_policy, request_origin)

def classify_xss_mitigation(parsed_policy: MultiValues) -> str: ...
def classify_tls_enforcement(parsed_policy: MultiValues) -> str: ...
def classify_frame_ancestors(parsed_policy: MultiValues, request_origin: str) -> str: ...
def classify_form_action(parsed_policy: MultiValues, request_origin: str) -> str: ...
```

#### ✅ AFTER: Object-Based Approach (Parameters Set Once)

```python
class CspParser:
    def __init__(self, csp_header_value: str, request_origin: str) -> None:
        self._parsed_policy = parse_content_security_policy(csp_header_value)
        self._request_origin = request_origin

    def classify_xss_mitigation(self) -> str:
        script_sources = self._get_script_sources()  # uses self._parsed_policy
        ...

    def classify_tls_enforcement(self) -> str:
        return self._parsed_policy.get("upgrade-insecure-requests") is not None

    def classify_frame_ancestors(self) -> str:
        frame_ancestors = self._parsed_policy.get("frame-ancestors")
        if not self._are_all_sources_self_origin(frame_ancestors, self._request_origin):
            ...

    def classify_form_action(self) -> str:
        ...

# Usage: create once, call methods without repeating parameters
def analyze_csp(header_value: str, request_origin: str) -> None:
    parser = CspParser(header_value, request_origin)  # set parameters ONCE
    parser.classify_xss_mitigation()
    parser.classify_tls_enforcement()
    parser.classify_frame_ancestors()
    parser.classify_form_action()
```

#### Key Benefits

1. **DRY**: `parsed_policy` and `request_origin` are set once in `__init__`
2. **Cleaner API**: `parser.classify_xss_mitigation()` vs `classify_xss_mitigation(parsed_policy)`
3. **Maintainability**: Add a new common parameter by updating `__init__` only
4. **Encapsulation**: The object owns its data and behavior together

---

## #011 Don't hide errors

When errors occur, do not hide them. The default should be to raise or re-raise. If errors are expected and unavoidable, at minimum log a warning.

**Example:**

```python
# ❌ BAD - silently swallowing errors
for tool_item in agent_tools:
    try:
        classifications = classify_single_tool(provider, tool_item)
    except Exception:
        continue  # ❌ Error is lost entirely

# ✅ GOOD - log and re-raise, or log warning if truly non-fatal
for tool_item in agent_tools:
    try:
        classifications = classify_single_tool(provider, tool_item)
    except ClassificationError as e:
        logger.warning(f"Failed to classify tool {tool_item.name!r}: {e}")
        raise
```

---

## #012 No hard coded configuration

Avoid hard coded configuration values. Prefer reading from environment variables or a config object.

**Example:**

```python
# ❌ BAD
MAX_FLOWS = 100  # Hard-coded constant

# ✅ GOOD - from environment variable with a default
import os
max_flows = int(os.environ.get("MAX_FLOWS", "100"))

# ✅ GOOD - from a config dataclass
from dataclasses import dataclass, field
import os

@dataclass
class AppConfig:
    max_flows: int = field(
        default_factory=lambda: int(os.environ.get("MAX_FLOWS", "100"))
    )
```

---

## #013 Limit indentation

Code with more than 3 levels of indentation is hard to read and understand. Extract inner logic into well-named helper methods.

**Example:**

```python
# ❌ BAD - 3+ nested loops with inline logic
def generate_three_step_flows(
    risk_context: RiskBuckets,
    agent_item: LocatedItem,
) -> list[RiskFlow]:
    flows = []
    three_step_count = 0
    for entry_risk in risk_context.entry_risks:
        for middle_risk in risk_context.middle_risks:
            for attack_risk in risk_context.attack_risks:  # ❌ 3+ levels
                three_step_count += 1
                if three_step_count > MAX_FLOWS:
                    return flows
                flows.append(
                    RiskFlow(entry_risk, middle_risk, attack_risk)
                )
    return flows

# ✅ GOOD - extract inner logic
def generate_three_step_flows(
    risk_context: RiskBuckets,
    agent_item: LocatedItem,
) -> list[RiskFlow]:
    flows = []
    for entry_risk in risk_context.entry_risks:
        for middle_risk in risk_context.middle_risks:
            new_flows = self._collect_attack_flows(entry_risk, middle_risk, risk_context)
            flows.extend(new_flows)
            if len(flows) >= MAX_FLOWS:
                return flows
    return flows
```

---

## #014 Provide error details

When raising an error, include the values that caused it. A stack trace tells you where the error occurred, but not what values triggered it.

**Example:**

```python
# ❌ BAD - no context about which risk tag was invalid
def get_flow_tag(exit_point_risk: ToolRisk) -> str:
    match exit_point_risk.risk_tag:
        case "dangerous_action":
            return "destructive-flow"
        case "external_write":
            return "exfiltration-flow"
        case _:
            raise ValueError("got invalid risk tag")  # ❌ Which tag?

# ✅ GOOD
def get_flow_tag(exit_point_risk: ToolRisk) -> str:
    match exit_point_risk.risk_tag:
        case "dangerous_action":
            return "destructive-flow"
        case "external_write":
            return "exfiltration-flow"
        case _:
            raise ValueError(f"got invalid risk tag: {exit_point_risk.risk_tag!r}")
```

---

## #015 Boolean condition

A boolean variable is a gift for use directly in a condition. Never compare a boolean to `True` or `False`.

**Example:**

```python
# ❌ BAD
is_enabled_logs = po_with_id.export_config.log_all_events
if is_enabled_logs == False:
    return Response()

# ✅ GOOD
if not po_with_id.export_config.log_all_events:
    return Response()
```

---

## #016 Initialize nested collections

When working with objects loaded from persistent storage (files, databases, APIs), always check and initialize each nested collection field individually, even if the parent object exists. You cannot assume that a non-None parent has its child collections initialized.

**Example:**

```python
# ❌ BAD - assumes nested dicts are present
def update_user_data(users: dict[str, dict], user_id: str) -> None:
    if user_id not in users:
        users[user_id] = {"preferences": {}, "orders": {}}

    # DANGER: if loaded from file, "preferences" or "orders" key may be missing
    users[user_id]["preferences"]["theme"] = "dark"

# ✅ GOOD - defensively initialize each nested level
def update_user_data(users: dict[str, dict], user_id: str) -> None:
    if user_id not in users:
        users[user_id] = {}

    user = users[user_id]
    if "preferences" not in user:
        user["preferences"] = {}
    if "orders" not in user:
        user["orders"] = {}

    user["preferences"]["theme"] = "dark"  # Safe
```

---

## #017 Use type hints - Avoid generic types

Always annotate function signatures and class attributes with type hints. Prefer specific, domain-named types over generic ones like `Any`, `dict`, or `list`. Use `TypedDict`, `dataclasses`, or `Pydantic` models to give structure to data.

**Example:**

```python
# ❌ BAD - no type hints, generic types
def process_data(key, obj):
    cache.set(key, obj)

def get_tools(agent) -> list:
    ...

data: dict = {}

# ✅ GOOD - specific type hints
def process_jwt_token(
    jwt_by_cookie_id: str,
    token_payload: JwtPayload,
) -> None:
    cache.set(jwt_by_cookie_id, token_payload)

def get_tools(agent: AgentItem) -> list[ToolItem]:
    ...

user_sessions: dict[str, UserSession] = {}
```

```python
# ❌ BAD - using Any when a better type exists
from typing import Any

def analyze(input_data: dict[str, Any]) -> Any:
    ...

# ✅ GOOD - use a dataclass or TypedDict
from dataclasses import dataclass

@dataclass
class AnalysisInput:
    session_id: str
    raw_values: list[str]
    threshold: float
```

---

## #018 Dataclass over complex dicts

Replace nested dicts (e.g., `dict[int, dict[str, int]]`) and parallel data structures with explicit dataclasses that have named fields. Dataclasses make relationships clear, improve type safety, and make code self-documenting.

**Example:**

```python
# ❌ BAD - opaque list of dicts
def calculate_entropy(session_id_values: list[str]) -> float:
    chunk_frequencies: list[dict[str, int]] = [
        {} for _ in range(self.max_length // self.tuple_length)
    ]
    # What does each index represent? What does each dict track?

# ✅ GOOD - named dataclass
from dataclasses import dataclass, field

@dataclass
class TuplePositionStats:
    position: int
    chunk_frequencies: dict[str, int] = field(default_factory=dict)
    total_count: int = 0

def calculate_entropy(session_id_values: list[str]) -> float:
    tuple_stats = [
        TuplePositionStats(position=i)
        for i in range(self.max_length // self.tuple_length)
    ]
    # Clear: we're tracking position-based statistics
```

---

## #019 Informative naming

Use informative, domain-specific names for parameters, variables, class attributes, and functions. Three concrete rules:

**Rule 1 — Every parameter must be at least two words.**
A single word rarely explains both what the value *is* and what *role* it plays. Two words force precision.

**Rule 2 — Never use generic or positional names.**
Banned names include: `i`, `j`, `k`, `index`, `key`, `value`, `data`, `obj`, `item`, `element`, `result`, `temp`, `x`, `y`.
These names say nothing about what the variable holds. Replace them with the domain concept.

**Rule 3 — For dicts, the name must reveal the key structure.**
Format: `<values>_by_<key>`. This makes it immediately clear what the key represents.

**Example — function parameters:**

```python
# ❌ BAD - single-word and generic parameter names
def process_data(key: str, obj: object) -> None:
    cache.set(key, obj)

def analyze(data: dict, index: int) -> str:
    ...

# ✅ GOOD - two-word, domain-specific names
def process_jwt_token(
    jwt_by_cookie_id: str,
    token_payload: JwtPayload,
) -> None:
    cache.set(jwt_by_cookie_id, token_payload)

def analyze(
    tool_config: dict,
    tool_position: int,
) -> str:
    ...
```

**Example — loop variables:**

```python
# ❌ BAD - single-letter loop variables, generic names
for i, item in enumerate(tools):
    process(i, item)

for key, value in config.items():
    results[key] = transform(value)

# ✅ GOOD - descriptive names that match the domain
for tool_index, tool_item in enumerate(tools):
    process(tool_index, tool_item)

for error_type, raw_score in config.items():
    scores_by_error_type[error_type] = transform(raw_score)
```

**Example — dict naming:**

```python
# ❌ BAD - name doesn't reveal what the key is
session_map: dict[str, UserSession] = {}
tool_data: dict[str, list[str]] = {}
class UserCache:
    data: dict[str, Any] = {}

# ✅ GOOD - name explains the key structure
sessions_by_session_id: dict[str, UserSession] = {}
argument_names_by_tool_name: dict[str, list[str]] = {}
class UserCache:
    user_sessions_by_session_id: dict[str, UserSession] = {}
```

---

## #020 Extract config once at method start

Always capture the configuration reference at the beginning of `enforce_rule()` or equivalent entry methods to avoid repeated deep attribute chains throughout the method.

```python
# ✅ GOOD - extract once
def enforce_rule(self, rule_input: RuleInput) -> list[LocatedRisk] | None:
    rule_config = self.rule_helper.get_po().aspm_info.aspm_rules.rule_your_rule
    if not rule_config.rule_enabled:
        return None
    threshold = rule_config.some_threshold
    ...

# ❌ BAD - repeated deep chains
def enforce_rule(self, rule_input: RuleInput) -> list[LocatedRisk] | None:
    if not self.rule_helper.get_po().aspm_info.aspm_rules.rule_your_rule.rule_enabled:
        return None
    threshold = self.rule_helper.get_po().aspm_info.aspm_rules.rule_your_rule.some_threshold
```

---

## #021 Let helper methods extract their own data

Do not pre-extract data in the caller and pass it alongside the full input. Instead, pass only the input and let the callee extract what it needs.

```python
# ✅ GOOD - helper extracts its own data
def enforce_rule(self, rule_input: RuleInput):
    return self._analyze_headers(rule_input)

def _analyze_headers(self, rule_input: RuleInput):
    headers = rule_input.transaction_parsed.parsed_response.http_headers_multi

# ❌ BAD - caller knows too much about callee's needs
def enforce_rule(self, rule_input: RuleInput):
    headers = rule_input.transaction_parsed.parsed_response.http_headers_multi
    return self._analyze_headers(rule_input, headers)  # Redundant — rule_input already has it
```

---

## #022 Follow the established pattern

Before writing new code, reference existing similar components as templates. They demonstrate the correct coupling patterns and conventions used across the codebase. Do not introduce a new pattern when an established one already exists.

---

## #023 Avoid unnecessary empty checks before loops

A `for` loop over an empty list or dict will simply not execute — no explicit `if not collection` guard is needed before it.

**Example:**

```python
# ❌ BAD - unnecessary guard
response_cookies = parse_response_cookies(rule_input.transaction_parsed)

if not response_cookies:  # ❌ Redundant — loop handles this
    return None

risks = []
for cookie_info in response_cookies:
    ...

# ✅ GOOD - loop handles empty collection naturally
response_cookies = parse_response_cookies(rule_input.transaction_parsed)

risks = []
for cookie_info in response_cookies:  # Won't execute if empty
    ...
```

---

## #024 Use datetime instead of epoch timestamps

Always use `datetime` objects instead of raw epoch integers (`int` or `float`). `datetime` is type-safe, timezone-aware, and provides rich comparison and arithmetic operations.

**Example:**

```python
import time
from datetime import datetime, timezone

# ❌ BAD - epoch integers
def is_expired(expiration_epoch: int) -> bool:
    now = int(time.time())
    return now > expiration_epoch

created_at = int(time.time())  # int — what unit? timezone? naive?

# ✅ GOOD - datetime objects
def is_expired(expiration_time: datetime) -> bool:
    return datetime.now(tz=timezone.utc) > expiration_time

created_at = datetime.now(tz=timezone.utc)  # self-documenting, timezone-aware
if created_at > some_other_time:
    ...
```

---

## #025 Return values and error handling

Use exceptions when reaching an unexpected branch indicates a programming bug. Return a meaningful empty value (`""`, `[]`, `None`) only when the default represents a valid "nothing found" state, not a bug.

**Example:**

```python
# ✅ Exception - default case is a BUG (should crash)
def get_severity_level(risk_type: str) -> tuple[str, int]:
    match risk_type:
        case "critical":
            return "high", 100
        case "minor":
            return "low", 10
        case _:
            raise ValueError(f"unknown risk type: {risk_type!r}")  # crash — it's a bug

# ✅ Empty return - default case is VALID (no risk found = normal)
def check_for_issue(value: str) -> str:
    match value:
        case "unsafe":
            return "Security risk detected"
        case "deprecated":
            return "Deprecated pattern found"
        case _:
            return ""  # No issue = valid state

# ❌ WRONG - returning empty string when an unknown code is a bug
def get_color(code: int) -> str:
    match code:
        case 1:
            return "red"
        case _:
            return ""  # ❌ Unknown code should raise, not silently return ""
```

---

## #026 Break from function ASAP

Exit functions as early as possible when conditions are not met. This reduces nesting and makes the happy path easier to read.

**Example:**

```python
# ❌ BAD - processes before checking threshold
def analyze_session(self, session_id_values: list[str], rule_config: RuleConfig) -> None:
    entropy = self._calculate_entropy(session_id_values, rule_config)
    if entropy < rule_config.min_entropy_threshold:
        risk = self._create_risk(entropy, rule_config)
        self.risks.append(risk)

# ✅ GOOD - exit early
def analyze_session(self, session_id_values: list[str], rule_config: RuleConfig) -> None:
    entropy = self._calculate_entropy(session_id_values, rule_config)
    if entropy >= rule_config.min_entropy_threshold:
        return  # Exit early — nothing to do

    risk = self._create_risk(entropy, rule_config)
    self.risks.append(risk)
```

---

## #027 Function length and extraction

Functions should be around 15–20 lines. When breaking up long functions, only extract helpers that will be reused or have high atomicity — meaning they make sense as standalone, single-purpose units. Do not over-extract single-use fragments just to shorten the parent.

**Example:**

```python
# ❌ BAD - over-extracted single-use helpers with no independent meaning
def enforce_rule(self, rule_input: RuleInput) -> list[LocatedRisk]:
    session_ids = self._parse_session_ids(rule_input)
    config = self._get_config()

    for session_id in session_ids:
        entropy = self._calculate_entropy(session_id)
        if self._is_below_threshold(entropy, config):   # Single-use, low atomicity
            self._add_risk(session_id, entropy)         # Single-use, low atomicity

# ✅ GOOD - only extract reusable, atomic helpers
def enforce_rule(self, rule_input: RuleInput) -> list[LocatedRisk]:
    rule_config = self.rule_helper.get_po().aspm_info.aspm_rules.rule_session_entropy
    if not rule_config.rule_enabled:
        return []

    session_ids = parse_session_ids(rule_input.transaction_parsed)  # Reusable
    risks = []

    for session_id in session_ids:
        entropy = self.entropy_calculator.calculate(session_id.values)  # Reusable, atomic

        if entropy >= rule_config.min_entropy_threshold:
            continue

        risk = self.rule_helper.create_risk(  # Reusable, atomic
            rule_config.risk_severity,
            f"Low entropy: {entropy:.2f}",
            session_id.name,
            rule_input,
        )
        risks.append(risk)

    return risks
```

---

## #028 Logging

Always use the `logging` module — never `print`. Use module-level loggers. Choose the appropriate level based on how frequently the message occurs.

**Log levels:**
- `DEBUG` — occurs very frequently (per item, per call)
- `INFO` — occurs per request or operation
- `WARNING` — unexpected but recoverable situation
- `ERROR` — failure that needs attention

**Example:**

```python
import logging

logger = logging.getLogger(__name__)

# ✅ GOOD
def create_risk(self, ...) -> LocatedRisk:
    risk = LocatedRisk(...)
    logger.debug("risk created")  # Frequent → DEBUG
    return risk

def enforce_rule(self, rule_input: RuleInput) -> list[LocatedRisk]:
    logger.info("enforcing rule")  # Per-request → INFO
    ...

def process_batch(self) -> None:
    logger.info("batch processing completed")

# ❌ BAD
print("risk created")     # Never use print for logging
print(f"error: {e}")      # Use logger.error() instead
```

---

## #029 Prefer simple solutions

Always choose the simplest solution that solves the problem. Fewer lines, fewer abstractions, and built-in Python idioms are preferred over verbose manual implementations. Do not over-engineer.

**Example — use comprehensions and builtins:**

```python
# ❌ BAD - manual loop for something a comprehension handles in one line
def get_enabled_tools(tool_list: list[Tool]) -> list[Tool]:
    enabled_tools = []
    for tool_item in tool_list:
        if tool_item.is_enabled:
            enabled_tools.append(tool_item)
    return enabled_tools

# ✅ GOOD
def get_enabled_tools(tool_list: list[Tool]) -> list[Tool]:
    return [tool_item for tool_item in tool_list if tool_item.is_enabled]
```

```python
# ❌ BAD - manual search with a flag
tool_found = False
for tool_item in tool_list:
    if tool_item.tool_name == target_tool_name:
        tool_found = True
        break

# ✅ GOOD
tool_found = any(tool_item.tool_name == target_tool_name for tool_item in tool_list)
```

```python
# ❌ BAD - manual dict building with a conditional
error_labels = {}
for error_type, error_count in error_counts_by_type.items():
    if error_count > 0:
        error_labels[error_type] = f"{error_type}: {error_count}"

# ✅ GOOD
error_labels = {
    error_type: f"{error_type}: {error_count}"
    for error_type, error_count in error_counts_by_type.items()
    if error_count > 0
}
```

---

## #030 Avoid code duplication (DRY)

Do not copy-paste logic. If the same logic appears in two places, extract it into a shared function. Every piece of knowledge should have a single authoritative representation in the codebase.

**Example:**

```python
# ❌ BAD - identical logic duplicated for two languages
def process_hebrew_error(
    error_text: str,
    article_context: str,
) -> str:
    prompt = f"Fix this Hebrew text: {error_text}\nContext: {article_context}"
    response = self.llm_client.complete(prompt)
    return response.strip()

def process_arabic_error(
    error_text: str,
    article_context: str,
) -> str:
    prompt = f"Fix this Arabic text: {error_text}\nContext: {article_context}"
    response = self.llm_client.complete(prompt)
    return response.strip()

# ✅ GOOD - shared logic extracted once
def _process_language_error(
    language_name: str,
    error_text: str,
    article_context: str,
) -> str:
    prompt = f"Fix this {language_name} text: {error_text}\nContext: {article_context}"
    return self.llm_client.complete(prompt).strip()

def process_hebrew_error(
    error_text: str,
    article_context: str,
) -> str:
    return self._process_language_error("Hebrew", error_text, article_context)

def process_arabic_error(
    error_text: str,
    article_context: str,
) -> str:
    return self._process_language_error("Arabic", error_text, article_context)
```

**Signs you need to apply DRY:**
- You are about to copy-paste a block of code
- Two functions differ only in a single value or type name
- A bug fix needs to be applied in more than one place

---

## #031 Never use mutable default arguments

Mutable default arguments (`list`, `dict`, `set`) are created **once** when the function is defined and shared across every call. This is one of the most common silent bugs in Python.

**Example:**

```python
# ❌ BAD - the list is created once and mutated across all calls
def add_error(
    error_text: str,
    error_list: list[str] = [],
) -> list[str]:
    error_list.append(error_text)
    return error_list

add_error("missing entity")   # ["missing entity"]
add_error("wrong number")     # ["missing entity", "wrong number"] — WRONG, leaks across calls

# ✅ GOOD - use None and initialize inside the function
def add_error(
    error_text: str,
    error_list: list[str] | None = None,
) -> list[str]:
    if error_list is None:
        error_list = []
    error_list.append(error_text)
    return error_list
```

This applies equally to `dict`, `set`, and any other mutable type as a default.

---

## #032 Use context managers for resource management

Any resource that needs cleanup — files, database connections, HTTP sessions, locks — must be managed with a `with` statement. Never manually call `.close()` or `.release()`. If an exception is raised between open and close, manual cleanup is silently skipped.

**Example:**

```python
# ❌ BAD - file never closed if an exception occurs
def save_results(
    output_file_path: str,
    result_lines: list[str],
) -> None:
    file_handle = open(output_file_path, "w", encoding="utf-8")
    for result_line in result_lines:
        file_handle.write(result_line + "\n")
    file_handle.close()

# ✅ GOOD - context manager guarantees cleanup
def save_results(
    output_file_path: str,
    result_lines: list[str],
) -> None:
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for result_line in result_lines:
            output_file.write(result_line + "\n")
```

```python
# ❌ BAD - lock never released if exception occurs mid-update
def update_cache(
    cache_key: str,
    cache_value: str,
) -> None:
    self._lock.acquire()
    self._cache[cache_key] = cache_value
    self._lock.release()

# ✅ GOOD
def update_cache(
    cache_key: str,
    cache_value: str,
) -> None:
    with self._lock:
        self._cache[cache_key] = cache_value
```

---

## #033 Catch specific exceptions

Never use a bare `except:` or a broad `except Exception:`. Catch only the specific exception types you know how to handle. Broad catches hide programming bugs and swallow signals like `KeyboardInterrupt` and `SystemExit`.

**Example:**

```python
# ❌ BAD - catches everything including bugs and system signals
try:
    llm_response = self._llm_client.complete(prompt_text)
except:
    return None

# ❌ BAD - still too broad, hides programming errors
try:
    llm_response = self._llm_client.complete(prompt_text)
except Exception:
    return None

# ✅ GOOD - catch only the errors you expect and can handle
try:
    llm_response = self._llm_client.complete(prompt_text)
except ApiRateLimitError as rate_limit_error:
    logger.warning(f"Rate limit hit, retrying: {rate_limit_error}")
    raise
except ApiConnectionError as connection_error:
    logger.error(f"LLM API unreachable: {connection_error}")
    raise
```

If you genuinely need to catch broadly (e.g., a top-level error boundary), log the full traceback and re-raise or convert to a domain error — never silently swallow.

---

## #034 Use Enum instead of string constants

A group of related named values should be an `Enum`, not a set of loose string constants. Raw strings are not type-safe: typos are silent, values are not discoverable, and the type annotation `str` gives no information about valid values.

**Example:**

```python
# ❌ BAD - loose strings, no validation, typos are undetected
ERROR_TYPE_ENTITY = "entity_swap"
ERROR_TYPE_NUMERIC = "numeric_error"
ERROR_TYPE_NEGATION = "negation"

def classify_error(error_type: str) -> str:  # accepts any string
    if error_type == "entity_swap":
        ...
    elif error_type == "nuumeric_error":  # typo — silently wrong
        ...

# ✅ GOOD - Enum: type-safe, IDE-completable, exhaustively matchable
from enum import Enum

class ErrorType(str, Enum):
    ENTITY_SWAP = "entity_swap"
    NUMERIC_ERROR = "numeric_error"
    NEGATION = "negation"

def classify_error(error_type: ErrorType) -> str:  # only valid values accepted
    match error_type:
        case ErrorType.ENTITY_SWAP:
            return "entity was swapped"
        case ErrorType.NUMERIC_ERROR:
            return "numeric value is wrong"
        case _:
            raise ValueError(f"Unhandled error type: {error_type!r}")
```

Using `class ErrorType(str, Enum)` keeps values as plain strings for JSON serialization compatibility.

---

## #035 Dependency injection

Never construct dependencies (LLM clients, database connections, file paths, API clients) inside `__init__`. Accept them as constructor parameters. Code that builds its own dependencies cannot be tested, cannot be swapped, and couples the class to one specific implementation.

**Example:**

```python
# ❌ BAD - class builds its own dependencies, impossible to test
class CorrectnessEvaluator:
    def __init__(self) -> None:
        self._llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._results_path = "/data/results.json"

# ✅ GOOD - dependencies are injected from outside
class CorrectnessEvaluator:
    def __init__(
        self,
        llm_client: LlmClientInterface,
        results_file_path: str,
    ) -> None:
        self._llm_client = llm_client
        self._results_file_path = results_file_path

# Production
evaluator = CorrectnessEvaluator(
    llm_client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    results_file_path="/data/results.json",
)

# Tests — swap with a mock, no environment needed
evaluator = CorrectnessEvaluator(
    llm_client=MockLlmClient(fixed_response="yes"),
    results_file_path="/tmp/test_results.json",
)

---

## #036 All imports at the top of the file

All `import` and `from … import` statements must appear at the very top of the file, after the module docstring (if any) and before any other code. Never place imports inside functions, methods, or conditional blocks.

**Order (PEP 8):**
1. Standard library imports
2. Third-party library imports
3. Local / project imports

Each group separated by a blank line.

**Example:**

```python
# ❌ BAD - imports scattered through the file
def load_model(model_path: str):
    import torch  # ❌ import inside function
    return torch.load(model_path)

def evaluate(summary_text: str):
    from transformers import pipeline  # ❌ import inside function
    ...

if FEATURE_ENABLED:
    import optional_module  # ❌ conditional import

# ✅ GOOD - all imports at the top, grouped and ordered
import os
import json
from datetime import datetime

import torch
from transformers import pipeline
from openai import OpenAI

from summary_augmentation.config import WorkflowConfig
from summary_augmentation.utils import compute_similarity
```
