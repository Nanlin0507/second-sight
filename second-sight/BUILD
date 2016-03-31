# Description:
#   An Android app to assist the visually impaired.
#   Based on the Tensorflow camera demo app for Android.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_binary(
    name = "libtensorflow_demo.so",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]) + [":libpthread.so"],
    copts = [
        "-std=c++11",
        "-mfpu=neon",
        "-O2",
    ],
    linkopts = [
        "-landroid",
        "-ljnigraphics",
        "-llog",
        "-lm",
        "-z defs",
        "-s",
        "-Wl,--icf=all",  # Identical Code Folding
        "-Wl,--exclude-libs,ALL",  # Exclude syms in all libs from auto export
    ],
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = ["//tensorflow/core:android_tensorflow_lib"],
)

# This library only exists as a workaround to satisfy dependencies
# that declare -lpthread in their linkopts. Although Android supports
# pthreads, it does not provide it as a separate library.
cc_binary(
    name = "libpthread.so",
    srcs = [],
    linkopts = ["-shared"],
    tags = [
        "manual",
        "notap",
    ],
)

cc_library(
    name = "tensorflow_native_libs",
    srcs = [
        ":libpthread.so",
        ":libtensorflow_demo.so",
    ],
    tags = [
        "manual",
        "notap",
    ],
)

java_library(
  name="google-api-client-android",
  visibility = ["//visibility:public"],
  exports = [
    "@com_fasterxml_jackson_core_jackson_core//jar",
    "@com_google_api_client_google_api_client//jar",
    "@com_google_api_client_google_api_client_android//jar",
    "@com_google_code_findbugs_jsr305//jar",
    "@com_google_guava_guava_jdk5//jar",
    "@com_google_http_client_google_http_client//jar",
    "@com_google_http_client_google_http_client_android//jar",
    "@com_google_http_client_google_http_client_jackson2//jar",
    "@com_google_oauth_client_google_oauth_client//jar",
    "@commons_codec_commons_codec//jar",
    "@commons_logging_commons_logging//jar",
    "@org_apache_httpcomponents_httpcore//jar",
  ],
)

java_library(
  name="google-http-client-gson",
  visibility = ["//visibility:public"],
  exports = [
    "@com_google_code_findbugs_jsr305//jar",
    "@com_google_code_gson_gson//jar",
    "@com_google_http_client_google_http_client//jar",
    "@com_google_http_client_google_http_client_gson//jar",
    "@commons_codec_commons_codec//jar",
    "@commons_logging_commons_logging//jar",
    "@org_apache_httpcomponents_httpcore//jar",
  ],
)

android_binary(
    name = "second-sight",
    srcs = glob([
        "src/**/*.java",
    ]),
    assets = glob(["assets/**"]),
    assets_dir = "assets",
    custom_package = "com.martinbede.secondsight",
    inline_constants = 1,
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":tensorflow_native_libs",
        "google-http-client-gson",
        "@google_api_services_vision//jar",
        "google-api-client-android",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "bin/**",
            "gen/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
