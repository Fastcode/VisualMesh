/*
 * Copyright (C) 2017-2019 Trent Houliston <trent@houliston.me>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef VISUALMESH_UTILITY_VULKAN_COMPUTE_HPP
#define VISUALMESH_UTILITY_VULKAN_COMPUTE_HPP

#include <spirv/unified1/GLSL.std.450.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <spirv/unified1/spirv.hpp11>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

struct Program {
    struct Config {
        Config()
          : enable_int8(false)
          , enable_int16(false)
          , enable_int64(false)
          , enable_float16(false)
          , enable_float64(false)
          , enable_debug(false)
          , enable_glsl_extensions(false)
          , address_model(spv::AddressingModel::Logical)
          , memory_model(spv::MemoryModel::GLSL450) {}

        bool enable_int8;
        bool enable_int16;
        bool enable_int64;
        bool enable_float16;
        bool enable_float64;
        bool enable_debug;
        bool enable_glsl_extensions;
        spv::AddressingModel address_model;
        spv::MemoryModel memory_model;
    };

    //
    // Predefined IDs for commonly used types and constants
    //
    static constexpr uint32_t GLSL_IMPORT_ID = 1;

    std::vector<uint32_t> extensions;
    std::vector<uint32_t> entry_points;
    std::vector<uint32_t> strings;
    std::vector<uint32_t> names;
    std::vector<uint32_t> decorations;
    std::vector<uint32_t> types;
    std::vector<uint32_t> array_types;
    std::vector<uint32_t> pointer_types;
    std::vector<uint32_t> constants;
    std::vector<uint32_t> globals;
    std::vector<uint32_t> functions;

    Program(const Config& config = Config()) : id_generator(1), descriptor_sets(0), config(config) {
        // Enable use of OpenCL extension functions
        if (config.enable_glsl_extensions) {
            auto glsl_import             = encode_string("GLSL.std.450");
            std::vector<uint32_t> params = {GLSL_IMPORT_ID};
            params.insert(params.end(), glsl_import.begin(), glsl_import.end());
            operation(extensions, spv::Op::OpExtInstImport, params);
            id_generator++;
        }
    }

    std::vector<uint32_t> build() {
        std::vector<uint32_t> output;

        output.push_back(spv::MagicNumber);  // Magic header ID
        output.push_back(spv::Version);      // Version 1.3.0
        output.push_back(0);                 // Generator
        output.push_back(id_generator);      // Max ID in program (fill in later)
        output.push_back(0);                 // Schema

        // Uses OpTypeMatrix
        // The Shader capability implicitly declares this capability anyway, so we may as well define it explicitly.
        operation(output, spv::Op::OpCapability, {static_cast<uint32_t>(spv::Capability::Matrix)});

        // Uses the Shader Execution Model.
        operation(output, spv::Op::OpCapability, {static_cast<uint32_t>(spv::Capability::Shader)});

        // Uses extended image formats (like R8 = single 8-bit unsigned normalised integer channel)
        operation(output, spv::Op::OpCapability, {static_cast<uint32_t>(spv::Capability::StorageImageExtendedFormats)});

        // Uses variable pointers.
        // Each variable pointer must be confined to a single Block-decorated struct in the StorageBuffer storage class.
        operation(
          output, spv::Op::OpCapability, {static_cast<uint32_t>(spv::Capability::VariablePointersStorageBuffer)});

        // Uses OpTypeFloat to declare the 16-bit floating-point type.
        if (config.enable_int8) {
            operation(output, spv::Op::OpCapability, {static_cast<uint32_t>(spv::Capability::Int8)});
        }

        // Uses OpTypeFloat to declare the 16-bit floating-point type.
        if (config.enable_int16) {
            operation(output, spv::Op::OpCapability, {static_cast<uint32_t>(spv::Capability::Int16)});
        }

        // Uses OpTypeFloat to declare the 64-bit floating-point type.
        if (config.enable_int64) {
            operation(output, spv::Op::OpCapability, {static_cast<uint32_t>(spv::Capability::Int64)});
        }

        // Uses OpTypeFloat to declare the 16-bit floating-point type.
        if (config.enable_float16) {
            operation(output, spv::Op::OpCapability, {static_cast<uint32_t>(spv::Capability::Float16)});
        }

        // Uses OpTypeFloat to declare the 64-bit floating-point type.
        if (config.enable_float64) {
            operation(output, spv::Op::OpCapability, {static_cast<uint32_t>(spv::Capability::Float64)});
        }

        output.insert(output.end(), extensions.begin(), extensions.end());

        // OpMemoryModel (defaults to Logical GLSL450).
        operation(output,
                  spv::Op::OpMemoryModel,
                  {static_cast<uint32_t>(config.address_model), static_cast<uint32_t>(config.memory_model)});

        // Extensions must come after the memory model.
        // These two extensions are needed for using variable pointers as well as the StorageBuffer storage class.
        operation(extensions, spv::Op::OpExtension, {encode_string("SPV_KHR_storage_buffer_storage_class")});
        operation(extensions, spv::Op::OpExtension, {encode_string("SPV_KHR_variable_pointers")});

        // Append rest of program
        output.insert(output.end(), entry_points.begin(), entry_points.end());
        output.insert(output.end(), strings.begin(), strings.end());
        output.insert(output.end(), names.begin(), names.end());
        output.insert(output.end(), decorations.begin(), decorations.end());
        output.insert(output.end(), types.begin(), types.end());
        output.insert(output.end(), constants.begin(), constants.end());
        output.insert(output.end(), array_types.begin(), array_types.end());
        output.insert(output.end(), pointer_types.begin(), pointer_types.end());
        output.insert(output.end(), globals.begin(), globals.end());
        output.insert(output.end(), functions.begin(), functions.end());

        return output;
    }

    void add_source_line(const std::string& file, const uint32_t& line, const uint32_t& column) {
        if (config.enable_debug) {
            std::vector<uint32_t> string_op = encode_string(file);
            string_op.insert(string_op.begin(), 0);

            std::pair<bool, uint32_t> id = check_duplicate_declaration(strings, spv::Op::OpString, 1, string_op);
            if (!id.first) {
                string_op[0] = id.second;
                operation(strings, spv::Op::OpString, string_op);
            }
            operation(functions, spv::Op::OpLine, {id.second, line, column});
        }
    }

    uint32_t add_name(const uint32_t& target_id, const std::string& name) {
        if (config.enable_debug) {
            std::vector<uint32_t> name_op = encode_string(name);
            name_op.insert(name_op.begin(), target_id);
            operation(names, spv::Op::OpName, name_op);
        }
        return target_id;
    }

    uint32_t add_extension(const std::string& extension) {
        std::vector<uint32_t> params = {0};
        auto ext                     = encode_string(extension);
        params.insert(params.end(), ext.begin(), ext.end());

        std::pair<bool, uint32_t> id = check_duplicate_declaration(extensions, spv::Op::OpExtInstImport, 1, params);
        if (!id.first) {
            params[0] = id.second;
            operation(extensions, spv::Op::OpExtInstImport, params);
        }

        return id.second;
    }

    uint32_t add_type(const spv::Op& op, const std::initializer_list<uint32_t>& params) {
        std::vector<uint32_t> p = {0};
        p.insert(p.end(), params.begin(), params.end());

        std::pair<bool, uint32_t> id = check_duplicate_declaration(types, op, 1, p);
        if (!id.first) {
            p[0] = id.second;
            operation(types, op, p);
        }

        return id.second;
    }

    uint32_t add_pointer(const uint32_t& type, const spv::StorageClass& storage_class) {
        std::vector<uint32_t> params = {0, static_cast<uint32_t>(storage_class), type};
        std::pair<bool, uint32_t> id = check_duplicate_declaration(pointer_types, spv::Op::OpTypePointer, 1, params);
        if (!id.first) {
            params[0] = id.second;
            operation(pointer_types, spv::Op::OpTypePointer, params);
        }

        return id.second;
    }

    uint32_t add_vec_type(const spv::Op& type,
                          const std::initializer_list<uint32_t>& type_params,
                          const uint32_t& num_components) {
        std::vector<uint32_t> p = {0, add_type(type, type_params), num_components};

        std::pair<bool, uint32_t> id = check_duplicate_declaration(types, spv::Op::OpTypeVector, 1, p);
        if (!id.first) {
            p[0] = id.second;
            operation(types, spv::Op::OpTypeVector, p);
        }

        return id.second;
    }

    uint32_t add_mat_type(const uint32_t& col_type, const uint32_t cols) {
        std::vector<uint32_t> params = {0, col_type, cols};

        std::pair<bool, uint32_t> id = check_duplicate_declaration(types, spv::Op::OpTypeMatrix, 1, params);
        if (!id.first) {
            params[0] = id.second;
            operation(types, spv::Op::OpTypeMatrix, params);
        }

        return id.second;
    }

    uint32_t add_struct(const std::initializer_list<uint32_t>& members) {
        std::vector<uint32_t> params = {0};
        params.insert(params.end(), members.begin(), members.end());

        std::pair<bool, uint32_t> id = check_duplicate_declaration(array_types, spv::Op::OpTypeStruct, 1, params);
        if (!id.first) {
            params[0] = id.second;
            operation(array_types, spv::Op::OpTypeStruct, params);
        }

        return id.second;
    }

    uint32_t add_array_type(const uint32_t& type) {
        std::vector<uint32_t> params = {0, type};

        std::pair<bool, uint32_t> id = check_duplicate_declaration(array_types, spv::Op::OpTypeRuntimeArray, 1, params);
        if (!id.first) {
            params[0] = id.second;
            operation(array_types, spv::Op::OpTypeRuntimeArray, params);
        }

        return id.second;
    }

    uint32_t add_array_type(const uint32_t& type, const uint32_t& length) {
        std::vector<uint32_t> params = {0, type, length};

        std::pair<bool, uint32_t> id = check_duplicate_declaration(array_types, spv::Op::OpTypeArray, 1, params);
        if (!id.first) {
            params[0] = id.second;
            operation(array_types, spv::Op::OpTypeArray, params);
        }

        return id.second;
    }

    uint32_t add_constant(const uint32_t& type,
                          const std::vector<uint32_t>& params,
                          const uint32_t& word_size = 1,
                          const bool& is_spec_op    = false) {

        const spv::Op constant_op  = is_spec_op ? spv::Op::OpSpecConstant : spv::Op::OpConstant;
        const spv::Op composite_op = is_spec_op ? spv::Op::OpSpecConstantComposite : spv::Op::OpConstantComposite;
        spv::Op op                 = ((params.size() / word_size) > 1) ? composite_op : constant_op;

        std::vector<uint32_t> p = {type, 0};
        std::transform(params.begin(), params.end(), std::back_inserter(p), [&](uint32_t ui) -> uint32_t {
            if (params.size() > 1) { return add_constant(add_type(spv::Op::OpTypeInt, {32, 0}), {ui}, 1, is_spec_op); }
            else {
                return ui;
            }
        });
        std::pair<bool, uint32_t> id = check_duplicate_declaration(constants, op, 2, p);
        if (!id.first) {
            p[1] = id.second;
            operation(constants, op, p);
        }

        return id.second;
    }

    uint32_t add_constant(const uint32_t& type, const std::initializer_list<uint32_t>& params) {
        return add_constant(type, params, 1, false);
    }

    uint32_t add_constant(const uint32_t& type, const std::vector<float>& params, const bool& is_spec_op = false) {
        std::vector<uint32_t> p = {type, 0};
        std::transform(params.begin(), params.end(), std::back_inserter(p), [&](float f) -> uint32_t {
            uint32_t word;
            std::memcpy(&word, &f, sizeof(float));
            if (params.size() > 1) { return add_constant(add_type(spv::Op::OpTypeFloat, {32}), {word}, 1, is_spec_op); }
            else {
                return word;
            }
        });

        const spv::Op constant_op    = is_spec_op ? spv::Op::OpSpecConstant : spv::Op::OpConstant;
        const spv::Op composite_op   = is_spec_op ? spv::Op::OpSpecConstantComposite : spv::Op::OpConstantComposite;
        spv::Op op                   = (params.size() > 1) ? composite_op : constant_op;
        std::pair<bool, uint32_t> id = check_duplicate_declaration(constants, op, 2, p);
        if (!id.first) {
            p[1] = id.second;
            operation(constants, op, p);
        }

        return id.second;
    }

    uint32_t add_constant(const uint32_t& type, const std::initializer_list<float>& params) {
        return add_constant(type, params, false);
    }

    uint32_t add_constant(const uint32_t& type, const std::vector<double>& params, const bool& is_spec_op = false) {
        std::vector<uint32_t> p = {type, 0};
        std::transform(params.begin(), params.end(), std::back_inserter(p), [&](double d) -> uint32_t {
            uint64_t words;
            std::memcpy(&words, &d, sizeof(uint64_t));
            if (params.size() > 1) {
                return add_constant(add_type(spv::Op::OpTypeFloat, {64}),
                                    {static_cast<uint32_t>(words & 0x00000000FFFFFFFF),
                                     static_cast<uint32_t>((words >> 32) & 0x00000000FFFFFFFF)},
                                    2,
                                    is_spec_op);
            }
            else {
                return words;
            }
        });

        // If we are creating a vector (composite) type then p should be correct as it is.
        // However, if we are not creating a vector, then p will contain IDs rather than the raw literal value
        const spv::Op constant_op  = is_spec_op ? spv::Op::OpSpecConstant : spv::Op::OpConstant;
        const spv::Op composite_op = is_spec_op ? spv::Op::OpSpecConstantComposite : spv::Op::OpConstantComposite;
        spv::Op op                 = composite_op;
        if (params.size() == 1) {
            op       = constant_op;
            double d = *params.begin();
            uint64_t words;
            std::memcpy(&words, &d, sizeof(uint64_t));
            p.back() = static_cast<uint32_t>(words & 0x00000000FFFFFFFF);
            p.push_back(static_cast<uint32_t>((words >> 32) & 0x00000000FFFFFFFF));
        }

        std::pair<bool, uint32_t> id = check_duplicate_declaration(constants, op, 2, p);
        if (!id.first) {
            p[1] = id.second;
            operation(constants, op, p);
        }

        return id.second;
    }

    uint32_t add_constant(const uint32_t& type, const std::initializer_list<double>& params) {
        return add_constant(type, params, false);
    }

    uint32_t add_variable(const uint32_t& type,
                          const spv::StorageClass& storage_class,
                          const uint32_t& initialiser = std::numeric_limits<uint32_t>::max()) {

        uint32_t id                  = id_generator++;
        std::vector<uint32_t> params = {type, id, static_cast<uint32_t>(storage_class)};
        if (initialiser != std::numeric_limits<uint32_t>::max()) { params.push_back(initialiser); }
        switch (storage_class) {
            case spv::StorageClass::Function: operation(functions, spv::Op::OpVariable, params); break;
            case spv::StorageClass::StorageBuffer:
            default: operation(globals, spv::Op::OpVariable, params); break;
        }
        return id;
    }

    uint32_t load_variable(const uint32_t& ptr,
                           const uint32_t& type,
                           const spv::MemoryAccessMask& mem_access = spv::MemoryAccessMask::MaskNone,
                           const uint32_t& mem_access_operand      = 0) {

        uint32_t id                  = id_generator++;
        std::vector<uint32_t> params = {type, id, ptr};
        if (mem_access != spv::MemoryAccessMask::MaskNone) {
            params.push_back(static_cast<uint32_t>(mem_access));
            if (mem_access == spv::MemoryAccessMask::Aligned) { params.push_back(mem_access_operand); }
        }
        operation(functions, spv::Op::OpLoad, params);
        return id;
    }

    void store_variable(const uint32_t& ptr,
                        const uint32_t& data,
                        const spv::MemoryAccessMask& mem_access = spv::MemoryAccessMask::MaskNone,
                        const uint32_t& mem_access_operand      = 0) {

        std::vector<uint32_t> params = {ptr, data};
        if (mem_access != spv::MemoryAccessMask::MaskNone) {
            params.push_back(static_cast<uint32_t>(mem_access));
            if (mem_access == spv::MemoryAccessMask::Aligned) { params.push_back(mem_access_operand); }
        }
        operation(functions, spv::Op::OpStore, params);
    }

    uint32_t array_access(const uint32_t& ptr, const uint32_t& index, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpInBoundsPtrAccessChain, {type, id, ptr, index});
        return id;
    }

    uint32_t member_access(const uint32_t& ptr, const std::initializer_list<uint32_t>& indexes, const uint32_t& type) {
        uint32_t id                  = id_generator++;
        std::vector<uint32_t> params = {type, id, ptr};
        params.insert(params.end(), indexes.begin(), indexes.end());
        operation(functions, spv::Op::OpAccessChain, params);
        return id;
    }

    void add_decoration(const uint32_t& id,
                        const spv::Decoration& decoration,
                        const std::initializer_list<uint32_t>& operands) {
        std::vector<uint32_t> params = {id, static_cast<uint32_t>(decoration)};
        params.insert(params.end(), operands.begin(), operands.end());

        std::pair<bool, uint32_t> check = check_duplicate_declaration(decorations, spv::Op::OpDecorate, -1, params);
        if (!check.first) { operation(decorations, spv::Op::OpDecorate, params); }
    }

    void add_member_decoration(const uint32_t& struct_id,
                               const uint32_t& member_offset,
                               const spv::Decoration& decoration,
                               const std::initializer_list<uint32_t>& operands = {}) {
        std::vector<uint32_t> params = {struct_id, member_offset, static_cast<uint32_t>(decoration)};
        params.insert(params.end(), operands.begin(), operands.end());

        std::pair<bool, uint32_t> id = check_duplicate_declaration(decorations, spv::Op::OpMemberDecorate, -1, params);
        if (!id.first) { operation(decorations, spv::Op::OpMemberDecorate, params); }
    }

    uint32_t add_decoration_group(const spv::Decoration& decoration,
                                  const std::initializer_list<uint32_t>& operands = {}) {
        uint32_t id = id_generator++;
        // First create the decoration (the ID on the decoration needs to be a forward reference)
        add_decoration(id, decoration, operands);

        // Now create the decoration group
        operation(decorations, spv::Op::OpDecorationGroup, {id});

        return id;
    }

    void add_group_decoration(const uint32_t& group_id, const std::vector<uint32_t>& targets) {
        std::vector<uint32_t> params = {group_id};
        params.insert(params.end(), targets.begin(), targets.end());
        operation(decorations, spv::Op::OpGroupDecorate, params);
    }

    void add_import_decoration(const std::string& symbol, const uint32_t& symbol_id) {
        std::vector<uint32_t> params = {symbol_id, static_cast<uint32_t>(spv::Decoration::LinkageAttributes)};

        auto encoded_symbol = encode_string(symbol);
        params.insert(params.end(), encoded_symbol.begin(), encoded_symbol.end());

        params.push_back(static_cast<uint32_t>(spv::LinkageType::Import));

        operation(decorations, spv::Op::OpDecorate, params);
    }

    void add_export_decoration(const std::string& symbol, const uint32_t& symbol_id) {
        std::vector<uint32_t> params = {symbol_id, static_cast<uint32_t>(spv::Decoration::LinkageAttributes)};

        auto encoded_symbol = encode_string(symbol);
        params.insert(params.end(), encoded_symbol.begin(), encoded_symbol.end());

        params.push_back(static_cast<uint32_t>(spv::LinkageType::Export));

        operation(decorations, spv::Op::OpDecorate, params);
    }

    void add_builtin_decoration(const uint32_t& symbol_id, const spv::BuiltIn& builtin) {
        operation(decorations,
                  spv::Op::OpDecorate,
                  {symbol_id, static_cast<uint32_t>(spv::Decoration::BuiltIn), static_cast<uint32_t>(builtin)});
    }

    void create_descriptor_set(const std::vector<uint32_t>& members) {
        // Decorate each of the members with the appropriate descriptor set bindings.
        for (uint32_t i = 0; i < static_cast<uint32_t>(members.size()); ++i) {
            operation(decorations,
                      spv::Op::OpDecorate,
                      {members[i], static_cast<uint32_t>(spv::Decoration::DescriptorSet), descriptor_sets});
            operation(
              decorations, spv::Op::OpDecorate, {members[i], static_cast<uint32_t>(spv::Decoration::Binding), i});
        }

        // Update descriptor sets counter.
        ++descriptor_sets;
    }

    // Add the beginning of a function, including both function declaration and definition
    // params = {return type, arg type 1, arg type 2, ...., arg type N}
    std::vector<uint32_t> begin_function(
      const std::string& name,
      const std::initializer_list<uint32_t>& params,
      const spv::FunctionControlMask& func_control = spv::FunctionControlMask::MaskNone) {

        // Add function declaration.
        uint32_t declaration      = add_function_type(params);
        std::vector<uint32_t> ids = {id_generator++};

        // Add function definition
        operation(functions,
                  spv::Op::OpFunction,
                  {*(params.begin()), ids[0], static_cast<uint32_t>(func_control), declaration});
        add_name(ids[0], name);

        // Define the functions parameter list
        for (auto param = std::next(params.begin()); param != params.end(); param++) {
            ids.push_back(id_generator++);
            operation(functions, spv::Op::OpFunctionParameter, {*param, ids.back()});
        }

        // Add the functions start block label
        operation(functions, spv::Op::OpLabel, {id_generator++});

        return ids;
    }

    // Add the beginning of a function/entry point, including function declaration, function definition,
    // and global variable declarations
    void begin_entry_point(const std::string& name, const std::initializer_list<uint32_t>& interface) {
        // Add function declaration
        uint32_t declaration = add_function_type({add_type(spv::Op::OpTypeVoid, {})});
        uint32_t func_id     = id_generator++;

        // Declare the entry point.
        add_entry_point(func_id, encode_string(name), interface);

        // Add function definition
        operation(functions,
                  spv::Op::OpFunction,
                  {add_type(spv::Op::OpTypeVoid, {}),
                   func_id,
                   static_cast<uint32_t>(spv::FunctionControlMask::MaskNone),
                   declaration});

        // Add the functions start block label
        operation(functions, spv::Op::OpLabel, {id_generator++});
    }

    void return_function() {
        operation(functions, spv::Op::OpReturn, {});
    }

    void return_function(const uint32_t& value) {
        operation(functions, spv::Op::OpReturnValue, {value});
    }

    void end_function() {
        operation(functions, spv::Op::OpFunctionEnd, {});
    }

    uint32_t call_function(const uint32_t& func_id,
                           const uint32_t& return_type,
                           const std::initializer_list<uint32_t>& params) {

        uint32_t id             = id_generator++;
        std::vector<uint32_t> p = {return_type, id, func_id};
        p.insert(p.end(), params.begin(), params.end());
        operation(functions, spv::Op::OpFunctionCall, p.begin(), p.end());
        return id;
    }

    // Casting routines
    uint32_t cast_int_to_float(const uint32_t& i, const uint32_t& result_type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpConvertSToF, {result_type, id, i});
        return id;
    }

    uint32_t cast_uint_to_float(const uint32_t& u, const uint32_t& result_type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpConvertUToF, {result_type, id, u});
        return id;
    }

    uint32_t cast_float_to_int(const uint32_t& f, const uint32_t& result_type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpConvertFToS, {result_type, id, f});
        return id;
    }

    uint32_t cast_float_to_uint(const uint32_t& f, const uint32_t& result_type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpConvertFToU, {result_type, id, f});
        return id;
    }

    // Arithmetic routines
    uint32_t floor(const uint32_t& var, const uint32_t& ftype, const uint32_t& itype) {
        uint32_t id = id_generator++;

        if (config.enable_glsl_extensions) {
            operation(
              functions, spv::Op::OpExtInst, {ftype, id, 1, static_cast<uint32_t>(GLSLstd450::GLSLstd450Floor), var});
        }

        else {
            operation(functions, spv::Op::OpConvertFToS, {itype, id, var});

            uint32_t id1 = id;
            id           = id_generator++;
            operation(functions, spv::Op::OpConvertSToF, {ftype, id, id1});
        }

        return id;
    }

    uint32_t fmod(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        // GLSLstd450::GLSLstd450Modf has the wrong semantics so we would need to do the calculation manually
        // May as well just use the standard SPIR-V operation
        operation(functions, spv::Op::OpFMod, {type, id, u, v});

        return id;
    }

    uint32_t umod(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpUMod, {type, id, u, v});
        return id;
    }

    uint32_t smod(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpSMod, {type, id, u, v});
        return id;
    }

    uint32_t exp(const uint32_t& x, const uint32_t& type) {
        if (config.enable_glsl_extensions) {
            uint32_t id = id_generator++;
            operation(
              functions, spv::Op::OpExtInst, {type, id, 1, static_cast<uint32_t>(GLSLstd450::GLSLstd450Exp), x});
            return id;
        }

        else {
            throw std::system_error(std::make_error_code(std::errc::function_not_supported),
                                    "Exp function only supported when GLSL extensions are enabled");
        }
    }

    uint32_t cos(const uint32_t& theta, const uint32_t& type) {
        if (config.enable_glsl_extensions) {
            uint32_t id = id_generator++;
            operation(
              functions, spv::Op::OpExtInst, {type, id, 1, static_cast<uint32_t>(GLSLstd450::GLSLstd450Cos), theta});
            return id;
        }

        else {
            throw std::system_error(std::make_error_code(std::errc::function_not_supported),
                                    "Cos function only supported when GLSL extensions are enabled");
        }
    }

    uint32_t acos(const uint32_t& x, const uint32_t& type) {
        if (config.enable_glsl_extensions) {
            uint32_t id = id_generator++;
            operation(
              functions, spv::Op::OpExtInst, {type, id, 1, static_cast<uint32_t>(GLSLstd450::GLSLstd450Acos), x});
            return id;
        }

        else {
            throw std::system_error(std::make_error_code(std::errc::function_not_supported),
                                    "Acos function only supported when GLSL extensions are enabled");
        }
    }

    uint32_t sin(const uint32_t& theta, const uint32_t& type) {
        if (config.enable_glsl_extensions) {
            uint32_t id = id_generator++;
            operation(
              functions, spv::Op::OpExtInst, {type, id, 1, static_cast<uint32_t>(GLSLstd450::GLSLstd450Sin), theta});
            return id;
        }

        else {
            throw std::system_error(std::make_error_code(std::errc::function_not_supported),
                                    "Sin function only supported when GLSL extensions are enabled");
        }
    }

    uint32_t asin(const uint32_t& x, const uint32_t& type) {
        if (config.enable_glsl_extensions) {
            uint32_t id = id_generator++;
            operation(
              functions, spv::Op::OpExtInst, {type, id, 1, static_cast<uint32_t>(GLSLstd450::GLSLstd450Asin), x});
            return id;
        }

        else {
            throw std::system_error(std::make_error_code(std::errc::function_not_supported),
                                    "Asin function only supported when GLSL extensions are enabled");
        }
    }

    uint32_t tan(const uint32_t& theta, const uint32_t& type) {
        if (config.enable_glsl_extensions) {
            uint32_t id = id_generator++;
            operation(
              functions, spv::Op::OpExtInst, {type, id, 1, static_cast<uint32_t>(GLSLstd450::GLSLstd450Tan), theta});
            return id;
        }

        else {
            throw std::system_error(std::make_error_code(std::errc::function_not_supported),
                                    "Tan function only supported when GLSL extensions are enabled");
        }
    }

    uint32_t atan(const uint32_t& x, const uint32_t& type) {
        if (config.enable_glsl_extensions) {
            uint32_t id = id_generator++;
            operation(
              functions, spv::Op::OpExtInst, {type, id, 1, static_cast<uint32_t>(GLSLstd450::GLSLstd450Atan), x});
            return id;
        }

        else {
            throw std::system_error(std::make_error_code(std::errc::function_not_supported),
                                    "Atan function only supported when GLSL extensions are enabled");
        }
    }

    uint32_t rsqrt(const uint32_t& x, const uint32_t& type) {
        if (config.enable_glsl_extensions) {
            uint32_t id = id_generator++;
            operation(functions,
                      spv::Op::OpExtInst,
                      {type, id, 1, static_cast<uint32_t>(GLSLstd450::GLSLstd450InverseSqrt), x});
            return id;
        }

        else {
            throw std::system_error(std::make_error_code(std::errc::function_not_supported),
                                    "InverseSqrt(rsqrt) function only supported when GLSL extensions are enabled");
        }
    }

    uint32_t dot(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpDot, {type, id, u, v});
        return id;
    }

    uint32_t outer(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpOuterProduct, {type, id, u, v});
        return id;
    }

    uint32_t iadd(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpIAdd, {type, id, u, v});
        return id;
    }

    uint32_t fadd(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpFAdd, {type, id, u, v});
        return id;
    }

    uint32_t isub(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpISub, {type, id, u, v});
        return id;
    }

    uint32_t fsub(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpFSub, {type, id, u, v});
        return id;
    }

    uint32_t imul(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpIMul, {type, id, u, v});
        return id;
    }

    uint32_t fmul(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpFMul, {type, id, u, v});
        return id;
    }

    uint32_t vmul(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpVectorTimesScalar, {type, id, u, v});
        return id;
    }

    uint32_t mmul(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpMatrixTimesScalar, {type, id, u, v});
        return id;
    }

    uint32_t vmmul(const uint32_t& v, const uint32_t& A, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpVectorTimesMatrix, {type, id, v, A});
        return id;
    }

    uint32_t mvmul(const uint32_t& A, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpMatrixTimesVector, {type, id, A, v});
        return id;
    }

    uint32_t mmmul(const uint32_t& A, const uint32_t& B, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpMatrixTimesVector, {type, id, A, B});
        return id;
    }

    uint32_t udiv(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpUDiv, {type, id, u, v});
        return id;
    }

    uint32_t sdiv(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpSDiv, {type, id, u, v});
        return id;
    }

    uint32_t fdiv(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpFDiv, {type, id, u, v});
        return id;
    }

    uint32_t srem(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpSRem, {type, id, u, v});
        return id;
    }

    uint32_t frem(const uint32_t& u, const uint32_t& v, const uint32_t& type) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpFRem, {type, id, u, v});
        return id;
    }

    // Vector routines
    uint32_t swizzle(const uint32_t& vec, const uint32_t& type, const std::initializer_list<uint32_t>& order) {
        uint32_t id                  = id_generator++;
        std::vector<uint32_t> params = {type, id, vec, vec};
        params.insert(params.end(), order.begin(), order.end());
        operation(functions, spv::Op::OpVectorShuffle, params);
        return id;
    }

    uint32_t swizzle(const uint32_t& u,
                     const uint32_t& v,
                     const uint32_t& type,
                     const std::initializer_list<uint32_t>& order) {
        uint32_t id                  = id_generator++;
        std::vector<uint32_t> params = {type, id, u, v};
        params.insert(params.end(), order.begin(), order.end());
        operation(functions, spv::Op::OpVectorShuffle, params);
        return id;
    }

    uint32_t create_vector(const uint32_t& return_type, const std::vector<uint32_t>& components) {
        uint32_t id                  = id_generator++;
        std::vector<uint32_t> params = {return_type, id};
        params.insert(params.end(), components.begin(), components.end());
        operation(functions, spv::Op::OpCompositeConstruct, params);
        return id;
    }

    uint32_t vector_component(const uint32_t& type, const uint32_t& vec, const uint32_t& component) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpCompositeExtract, {type, id, vec, component});
        return id;
    }

    uint32_t vector_modify(const uint32_t& type, const uint32_t& vec, const uint32_t& index, const uint32_t& object) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpCompositeInsert, {type, id, object, vec, index});
        return id;
    }

    uint32_t concatenate_uvec2(const uint32_t& u, const uint32_t& v) {
        uint32_t id = id_generator++;
        operation(
          functions, spv::Op::OpVectorShuffle, {add_vec_type(spv::Op::OpTypeInt, {32, 0}, 4), id, u, v, 0, 1, 2, 3});
        return id;
    }

    uint32_t concatenate_fvec2(const uint32_t& u, const uint32_t& v) {
        uint32_t id = id_generator++;
        operation(
          functions, spv::Op::OpVectorShuffle, {add_vec_type(spv::Op::OpTypeFloat, {32}, 4), id, u, v, 0, 1, 2, 3});
        return id;
    }

    // Image routines
    uint32_t add_image_type(const uint32_t& pixel_type,
                            const spv::Dim& dimension,
                            const bool& is_depth_image,
                            const bool& is_arrayed,
                            const bool& enable_multisampling,
                            const bool& with_sampler,
                            const spv::ImageFormat& image_format) {

        if (dimension == spv::Dim::SubpassData) {
            throw std::system_error(std::make_error_code(std::errc::function_not_supported),
                                    "Subpass data is not supported");
        }

        std::vector<uint32_t> params = {0,
                                        pixel_type,
                                        static_cast<uint32_t>(dimension),
                                        is_depth_image ? uint32_t(1) : uint32_t(0),
                                        is_arrayed ? uint32_t(1) : uint32_t(0),
                                        enable_multisampling ? uint32_t(1) : uint32_t(0),
                                        with_sampler ? uint32_t(1) : uint32_t(2),
                                        static_cast<uint32_t>(image_format)};

        std::pair<bool, uint32_t> id = check_duplicate_declaration(types, spv::Op::OpTypeImage, 1, params);
        if (!id.first) {
            params[0] = id.second;
            operation(types, spv::Op::OpTypeImage, params);
        }

        return id.second;
    }

    uint32_t add_sampler_type() {
        // First add the sampler type
        std::pair<bool, uint32_t> id = check_duplicate_declaration(types, spv::Op::OpTypeSampler, 1, {0});
        if (!id.first) { operation(types, spv::Op::OpTypeSampler, {id.second}); }
        return id.second;
    }

    uint32_t add_sampled_image_type(const uint32_t& image_id) {
        std::vector<uint32_t> params = {0, image_id};
        std::pair<bool, uint32_t> id = check_duplicate_declaration(types, spv::Op::OpTypeSampledImage, 1, params);
        if (!id.first) {
            params[0] = id.second;
            operation(types, spv::Op::OpTypeSampledImage, params);
        }

        return id.second;
    }

    uint32_t sample_image(const uint32_t& image_id,
                          const uint32_t& sampler_id,
                          const uint32_t& sampled_image_id,
                          const uint32_t& coordinates,
                          const uint32_t& pixel_type) {

        uint32_t id1 = id_generator++;
        uint32_t id2 = id_generator++;
        operation(functions, spv::Op::OpSampledImage, {sampled_image_id, id1, image_id, sampler_id});
        operation(functions,
                  spv::Op::OpImageSampleExplicitLod,
                  {pixel_type,
                   id2,
                   id1,
                   coordinates,
                   static_cast<uint32_t>(spv::ImageOperandsMask::Lod),
                   add_constant(add_type(spv::Op::OpTypeFloat, {32}), {0.0f})});

        return id2;
    }

    // Conditional routines
    uint32_t select(const uint32_t& type,
                    const uint32_t& condition,
                    const uint32_t& true_value,
                    const uint32_t& false_value) {

        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpSelect, {type, id, condition, true_value, false_value});
        return id;
    }

    // Reserve an ID to be used later with end_if
    uint32_t begin_if() {
        return id_generator++;
    }

    void end_if(const uint32_t& branch_id) {
        operation(functions, spv::Op::OpLabel, {branch_id});
    }

    uint32_t feq(const uint32_t& u, const uint32_t& v) {
        uint32_t id = id_generator++;
        operation(functions, spv::Op::OpFOrdEqual, {add_type(spv::Op::OpTypeBool, {}), id, u, v});
        return id;
    }

    uint32_t fgeq(const uint32_t& u, const uint32_t& v, const uint32_t& components = 1) {
        uint32_t id = id_generator++;
        uint32_t type;
        if (components == 1) { type = add_type(spv::Op::OpTypeBool, {}); }
        else {
            type = add_vec_type(spv::Op::OpTypeBool, {}, components);
        }
        operation(functions, spv::Op::OpFOrdGreaterThan, {type, id, u, v});
        return id;
    }

    std::pair<uint32_t, uint32_t> conditional(const uint32_t& condition) {
        uint32_t id1 = id_generator++;
        uint32_t id2 = id_generator++;
        operation(functions, spv::Op::OpBranchConditional, {condition, id1, id2});
        return std::make_pair(id1, id2);
    }

    void start_branch(const uint32_t& id) {
        operation(functions, spv::Op::OpLabel, {id});
    }

    void end_branch(const uint32_t& merge_id) {
        operation(functions, spv::Op::OpBranch, {merge_id});
    }

    uint32_t phi(const uint32_t& return_type, const std::initializer_list<uint32_t>& branches) {
        uint32_t id                  = id_generator++;
        std::vector<uint32_t> params = {return_type, id};
        params.insert(params.end(), branches.begin(), branches.end());
        operation(functions, spv::Op::OpPhi, params);
        return id;
    }

private:
    uint32_t id_generator;
    uint32_t descriptor_sets;
    Config config;

    std::vector<uint32_t> encode_string(const std::string& str) {
        std::vector<uint32_t> result((str.size() + 4) >> 2, 0);
        std::memcpy(reinterpret_cast<char*>(result.data()), str.data(), str.size());
        return result;
    }

    template <typename It>
    void operation(std::vector<uint32_t>& program, const spv::Op& op, const It& begin, const It& end) {
        program.push_back((std::distance(begin, end) + 1) << 16 | static_cast<uint32_t>(op));
        program.insert(program.end(), begin, end);
    }

    void operation(std::vector<uint32_t>& program, const spv::Op& op, const std::vector<uint32_t>& params) {
        operation(program, op, params.begin(), params.end());
    }

    void operation(std::vector<uint32_t>& program, const spv::Op& op, const std::initializer_list<uint32_t>& params) {
        operation(program, op, params.begin(), params.end());
    }


    void add_entry_point(const uint32_t& entry_point_id,
                         const std::vector<uint32_t>& name,
                         const std::vector<uint32_t>& interface) {
        std::vector<uint32_t> entry_point = {static_cast<uint32_t>(spv::ExecutionModel::GLCompute), entry_point_id};
        entry_point.insert(entry_point.end(), name.begin(), name.end());
        entry_point.insert(entry_point.end(), interface.begin(), interface.end());
        operation(entry_points, spv::Op::OpEntryPoint, entry_point);
        operation(entry_points,
                  spv::Op::OpExecutionMode,
                  {entry_point_id, static_cast<uint32_t>(spv::ExecutionMode::LocalSize), 1, 1, 1});
    }

    uint32_t add_function_type(const std::initializer_list<uint32_t>& params) {
        std::vector<uint32_t> p = {0};
        p.insert(p.end(), params);

        std::pair<bool, uint32_t> func_id = check_duplicate_declaration(types, spv::Op::OpTypeFunction, 1, p);
        if (!func_id.first) {
            p[0] = func_id.second;
            operation(types, spv::Op::OpTypeFunction, p);
        }

        return func_id.second;
    }

    std::pair<bool, uint32_t> check_duplicate_declaration(const std::vector<uint32_t>& program,
                                                          const spv::Op& op,
                                                          const uint32_t& result_id_index,
                                                          const std::vector<uint32_t>& params) {

        uint32_t opcode = uint32_t(params.size() + 1) << 16 | static_cast<uint32_t>(op);

        if (program.size() > 0) {
            for (size_t i = 0; i < (program.size() - params.size() + 1); i++) {
                if (program[i] == opcode) {
                    uint32_t match_count = 0;

                    for (size_t j = 0, k = i + 1; j < params.size(); j++, k++) {
                        if ((j + 1) == result_id_index) { match_count++; }
                        else {
                            if (program[k] == params[j]) { match_count++; }
                        }
                    }

                    if (match_count == params.size()) { return std::make_pair(true, program[i + result_id_index]); }
                }
            }
        }

        return std::make_pair(false, id_generator++);
    }
};

#endif  // VISUALMESH_UTILITY_VULKAN_COMPUTE_HPP
