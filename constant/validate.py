from enum import Enum


class ValidateErrorTypeEnum(str, Enum):
    json_invalid = "json_invalid"
    missing = "missing"

    # type
    json_type = "json_type"
    arguments_type = "arguments_type"
    bool_type = "bool_type"
    bytes_type = "bytes_type"
    date_type = "date_type"
    datetime_type = "datetime_type"
    decimal_type = "decimal_type"
    dict_type = "dict_type"
    enum = "enum"
    int_type = "int_type"
    list_type = "list_type"
    set_type = "set_type"
    string_type = "string_type"
    time_delta_type = "time_delta_type"
    time_type = "time_type"
    tuple_type = "tuple_type"
    url_type = "url_type"
    uuid_type = "uuid_type"
    model_attributes_type = "model_attributes_type"
    float_type = "float_type"

    # string_error
    string_too_short = "string_too_short"
    string_too_long = "string_too_long"
    string_pattern_mismatch = "string_pattern_mismatch"
    string_unicode = "string_unicode"

    # pase_error
    int_parsing = "int_parsing"
    decimal_parsing = "decimal_parsing"
    bool_parsing = "bool_parsing"
    date_from_datetime_parsing = "date_from_datetime_parsing"
    datetime_from_date_parsing = "datetime_from_date_parsing"
    float_parsing = "float_parsing"
    int_parsing_size = "int_parsing_size"
    time_delta_parsing = "time_delta_parsing"
    time_parsing = "time_parsing"
    url_parsing = "url_parsing"
    uuid_parsing = "uuid_parsing"

    # value_error
    value_error = "value_error"
    greater_than = "greater_than"
    multiple_of = "multiple_of"
    date_future = "date_future"
    date_past = "date_past"
    datetime_future = "datetime_future"
    datetime_past = "datetime_past"
    decimal_max_digits = "decimal_max_digits"
    decimal_max_places = "decimal_max_places"
    decimal_whole_digits = "decimal_whole_digits"
    extra_forbidden = "extra_forbidden"
    finite_number = "finite_number"
    greater_than_equal = "greater_than_equal"
    int_from_float = "int_from_float"
    less_than = "less_than"
    less_than_equal = "less_than_equal"
    literal_error = "literal_error"
    none_required = "none_required"
    timezone_aware = "timezone_aware"
    timezone_naive = "timezone_naive"
    too_long = "too_long"  # 列表
    too_short = "too_short"
    url_scheme = "url_scheme"
    url_syntax_violation = "url_syntax_violation"
    url_too_long = "url_too_long"
    uuid_version = "uuid_version"
    regex_error = "regex_error"

    # bytes
    bytes_too_long = "bytes_too_long"
    bytes_too_short = "bytes_too_short"


DirectValidateErrorMsgTemplates = {
    ValidateErrorTypeEnum.json_invalid: ("请求体", "格式异常"),
}


# CN_ZH
ValidationErrorMsgTemplates = {
    ValidateErrorTypeEnum.missing: "缺少必填字段",
    # type
    ValidateErrorTypeEnum.json_type: "不是合法的JSON格式",
    ValidateErrorTypeEnum.arguments_type: "参数类型错误",
    ValidateErrorTypeEnum.bool_type: "不是有效的布尔值",
    ValidateErrorTypeEnum.bytes_type: "不是有效的字节",
    ValidateErrorTypeEnum.date_type: "不是有效的日期",
    ValidateErrorTypeEnum.datetime_type: "不是有效的日期时间",
    ValidateErrorTypeEnum.decimal_type: "不是有效的小数",
    ValidateErrorTypeEnum.dict_type: "不是有效的字典",
    ValidateErrorTypeEnum.enum: "不是有效的枚举值, 可选值: {expected}",
    ValidateErrorTypeEnum.int_type: "不是有效的整数",
    ValidateErrorTypeEnum.list_type: "不是有效的列表",
    ValidateErrorTypeEnum.set_type: "不是有效的集合",
    ValidateErrorTypeEnum.string_type: "不是有效的字符串",
    ValidateErrorTypeEnum.time_delta_type: "不是有效的时间间隔",
    ValidateErrorTypeEnum.time_type: "不是有效的时间",
    ValidateErrorTypeEnum.tuple_type: "不是有效的元组",
    ValidateErrorTypeEnum.url_type: "不是有效的URL",
    ValidateErrorTypeEnum.uuid_type: "不是有效的UUID",
    ValidateErrorTypeEnum.model_attributes_type: "类型错误",
    ValidateErrorTypeEnum.float_type: "不是有效的浮点数",
    # string_error
    ValidateErrorTypeEnum.string_too_short: "至少{min_length}个字符",
    ValidateErrorTypeEnum.string_too_long: "最多{max_length}个字符",
    ValidateErrorTypeEnum.string_pattern_mismatch: "格式不正确",
    ValidateErrorTypeEnum.string_unicode: "不能包含非Unicode字符",
    # pase_error
    ValidateErrorTypeEnum.int_parsing: "请输入正确的整数",
    ValidateErrorTypeEnum.bool_parsing: "请输入正确的布尔值",
    ValidateErrorTypeEnum.decimal_parsing: "请输入正确的小数",
    ValidateErrorTypeEnum.date_from_datetime_parsing: "请输入正确的日期格式",
    ValidateErrorTypeEnum.datetime_from_date_parsing: "请输入正确的日期时间格式",
    ValidateErrorTypeEnum.float_parsing: "请输入正确的浮点数",
    ValidateErrorTypeEnum.int_parsing_size: "整数位数过长",
    ValidateErrorTypeEnum.time_delta_parsing: "请输入正确的时间间隔",
    ValidateErrorTypeEnum.time_parsing: "请输入正确的时间格式",
    ValidateErrorTypeEnum.url_parsing: "请输入正确的URL格式",
    ValidateErrorTypeEnum.uuid_parsing: "请输入正确的UUID格式",
    # value_error
    ValidateErrorTypeEnum.value_error: "值错误",
    ValidateErrorTypeEnum.multiple_of: "必须是{multiple_of}的整数倍",
    ValidateErrorTypeEnum.date_future: "必须是未来日期",
    ValidateErrorTypeEnum.date_past: "必须是过去日期",
    ValidateErrorTypeEnum.datetime_future: "必须是未来日期时间",
    ValidateErrorTypeEnum.datetime_past: "必须是过去日期时间",
    ValidateErrorTypeEnum.decimal_max_digits: "总位数不能超过{max_digits}",
    ValidateErrorTypeEnum.decimal_max_places: "小数位数不能超过{max_places}",
    ValidateErrorTypeEnum.decimal_whole_digits: "整数位数不能超过{whole_digits}",
    ValidateErrorTypeEnum.extra_forbidden: "不允许的额外字段",
    ValidateErrorTypeEnum.finite_number: "必须是有限的数字",
    ValidateErrorTypeEnum.greater_than: "必须大于{gt}",
    ValidateErrorTypeEnum.greater_than_equal: "必须大于等于{ge}",
    ValidateErrorTypeEnum.int_from_float: "必须是规范的整数",
    ValidateErrorTypeEnum.less_than: "必须小于{lt}",
    ValidateErrorTypeEnum.less_than_equal: "必须小于等于{le}",
    ValidateErrorTypeEnum.literal_error: "不是有效的值, 可选: {expected}",
    ValidateErrorTypeEnum.none_required: "只能为空值",
    ValidateErrorTypeEnum.timezone_aware: "必须是带时区信息的日期时间",
    ValidateErrorTypeEnum.timezone_naive: "必须是不带时区信息的日期时间",
    ValidateErrorTypeEnum.too_long: "长度不能超过{max_length}",
    ValidateErrorTypeEnum.too_short: "长度不能少于{min_length}",
    ValidateErrorTypeEnum.url_scheme: "必须是{expected_schemes}协议的URL",
    ValidateErrorTypeEnum.url_syntax_violation: "URL格式不正确",
    ValidateErrorTypeEnum.url_too_long: "URL长度不能超过{max_length}字节",
    ValidateErrorTypeEnum.uuid_version: "UUID版本不正确, 请使用版本{expected_version}",
    # bytes
    ValidateErrorTypeEnum.bytes_too_long: "字节数不能超过{max_length}字节",
    ValidateErrorTypeEnum.bytes_too_short: "字节数不能少于{min_length}字节",
}
