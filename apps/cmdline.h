#pragma once

#include <gdt/math/vec.h>

#include <string>
#include <cstring>
#include <iostream>

#include "args.hxx"

namespace args_impl {

struct Vec3fReader
{
  void operator()(const std::string &name, const std::string &in, gdt::vec3f &destination)
  {
    const auto s = in.find_first_not_of('(');
    const auto e = in.find_last_not_of (')');
    auto value = in.substr(s, e - s+1);

    auto curr = value.substr(0, value.find_first_of(','));
    value = value.substr(value.find_first_of(',') + 1);

    destination.x = std::stod(curr);

    curr = value.substr(0, value.find_first_of(','));
    value = value.substr(value.find_first_of(',') + 1);

    destination.y = std::stod(curr);

    destination.z = std::stod(value);
  }
};

}

struct CmdArgsBase {
public:
  template<typename T, int N>
  gdt::vec_t<T, N> ValueFlagList2Vec(args::ValueFlagList<float>& flag, const gdt::vec_t<T, N>& value) 
  {
    gdt::vec_t<T, N> ret = value;
    if (flag) {
      const auto& f = args::get(flag);
      for (int i = 0; i < f.size(); ++i) ret[i] = f[i];
    }
    return ret;
  }

  // Getter for the old Combo() API: "item1\0item2\0item3\0"
  static int Items_Count(const void* data)
  {
    int items_count = 0;
    const char* p = (const char*)data;
    while (*p)
    {
        p += strlen(p) + 1;
        items_count++;
    }
    return items_count;
  }

  // Getter for the old Combo() API: "item1\0item2\0item3\0"
  static bool Items_SingleStringGetter(const void* data, int idx, const char** out_text)
  {
      const char* items_separated_by_zeros = (const char*)data;
      int items_count = 0;
      const char* p = items_separated_by_zeros;
      while (*p)
      {
          if (idx == items_count)
              break;
          p += strlen(p) + 1;
          items_count++;
      }
      if (!*p)
          return false;
      if (out_text)
          *out_text = p;
      return true;
  }

public:
  void exec(args::ArgumentParser& parser, int argc, char** argv)
  {
    try {
      parser.ParseCLI(argc, argv);
    }
    catch (args::Help) {
      std::cout << parser;
      throw std::runtime_error(parser.Description());
    }
    catch (args::ParseError e) {
      std::cerr << e.what() << std::endl;
      std::cerr << parser;
      throw e;
    }
    catch (args::ValidationError e) {
      std::cerr << e.what() << std::endl;
      std::cerr << parser;
      throw e;
    }
  }
};
