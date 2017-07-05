FILE(GLOB src
    "**.cpp"
    "**.hpp")

ADD_EXECUTABLE(visualmesh ${src})

# INSTALL(TARGETS visualmesh ARCHIVE DESTINATION lib)
