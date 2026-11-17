import re

PHONE_REGEX_CN = re.compile(r"^1[3-9]\d{9}$")

EMAIL_REGEX = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")

LETTER_DIGITS_ONLY_REGEX = re.compile(r"^[a-zA-Z0-9]+$")

LICENSE_NO_REGEX = re.compile(
    r"^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-HJ-NP-Z][A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z0-9挂学警港澳]$",
)

URI_REGEX = re.compile(r"http[s]?://[^/]+(/[^?#]+)?")
