---note to self: give up on static linking, just do it fully dynamic for xmake, its not worth fighting the build system
add_rules("mode.debug", "mode.release")
add_requireconfs("*", {configs = {shared = false}, system=false})
add_requires("zlib")
add_requires("freetype")
add_requires("openssl", "opengl", "glfw", "glew", "sfml")

add_files("deps/imgui/backends/imgui_impl_glfw.cpp")
add_files("deps/imgui/backends/imgui_impl_opengl3.cpp")
add_files("deps/imgui/misc/freetype/imgui_freetype.cpp")
add_files("deps/imgui/misc/cpp/imgui_stdlib.cpp")
add_files("deps/imgui/imgui.cpp")
add_files("deps/imgui/imgui_draw.cpp")
add_files("deps/imgui/imgui_tables.cpp")
add_files("deps/imgui/imgui_widgets.cpp")
add_files("deps/libfastcl/fastcl/cl.cpp")
add_files("deps/toolkit/base_serialisables.cpp")
add_files("deps/toolkit/clipboard.cpp")
add_files("deps/toolkit/clock.cpp")
add_files("deps/toolkit/fs_helpers.cpp")
add_files("deps/toolkit/opencl.cpp")
add_files("deps/toolkit/render_window.cpp")
add_files("deps/toolkit/render_window_glfw.cpp")
add_files("deps/toolkit/stacktrace.cpp")
add_files("deps/toolkit/texture.cpp")
add_includedirs("./deps")
add_includedirs("./deps/imgui")
set_languages("c99", "cxx23")
add_defines("IMGUI_IMPL_OPENGL_LOADER_GLEW",
"SUBPIXEL_FONT_RENDERING",
"SFML_STATIC",
"GLEW_STATIC",
"GRAPHITE2_STATIC",
"CL_TARGET_OPENCL_VERSION=220",
"IMGUI_ENABLE_FREETYPE",
"NO_SERIALISE_RATELIMIT",
"FAST_CL",
"NO_OPENCL_SCREEN")

add_packages("openssl", "opengl", "glfw", "glew", "freetype", "sfml")

set_optimize("fastest")

add_links("imm32")
add_links("tbb12.dll")
add_links("dbgeng")

if is_plat("mingw") then
    add_ldflags("-static -static-libstdc++")
    add_cxflags("-mwindows")
    add_ldflags("-mwindows")
end

target("NumericalSim")
    set_kind("binary")
    add_files("main.cpp")
    add_files("*.cpp")