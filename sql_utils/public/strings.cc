/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sql_utils/public/strings.h"

#include <ctype.h>

#include <iterator>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"

namespace bigquery_ml_utils {

// Digit conversion.
static char hex_char[] = "0123456789abcdef";

// ----------------------------------------------------------------------
// CEscape()
// CHexEscape()
// Utf8SafeCEscape()
// Utf8SafeCHexEscape()
//    Escapes 'src' using C-style escape sequences.  This is useful for
//    preparing query flags.  The 'Hex' version uses hexadecimal rather than
//    octal sequences.  The 'Utf8Safe' version does not touch UTF-8 bytes.
//
//    Escaped chars: \n, \r, \t, ", ', \, and !ascii_isprint().
//
// COPIED FROM strings/escaping.cc, with unnecessary modes removed, and with
// the escape_quote_char feature added.
//
// If escape_quote_char is non-zero, only escape the quote character
// (from '"`) that matches escape_quote_char.
// This allows writing "ab'cd" or 'ab"cd' or `ab"cd` without extra escaping.
// ----------------------------------------------------------------------
static std::string CEscapeInternal(absl::string_view src, bool utf8_safe,
                                   char escape_quote_char) {
  std::string dest;
  bool last_hex_escape = false;  // true if last output char was \xNN.

  for (const char* p = src.begin(); p < src.end(); ++p) {
    unsigned char c = *p;
    bool is_hex_escape = false;
    switch (c) {
      case '\n': dest.append("\\" "n"); break;
      case '\r': dest.append("\\" "r"); break;
      case '\t': dest.append("\\" "t"); break;
      case '\\': dest.append("\\" "\\"); break;

      case '\'':
      case '\"':
      case '`':
        // Escape only quote chars that match escape_quote_char.
        if (escape_quote_char == 0 || c == escape_quote_char) {
          dest.push_back('\\');
        }
        dest.push_back(c);
        break;

      default:
        // Note that if we emit \xNN and the src character after that is a hex
        // digit then that digit must be escaped too to prevent it being
        // interpreted as part of the character code by C.
        if ((!utf8_safe || c < 0x80) &&
            (!absl::ascii_isprint(c) ||
             (last_hex_escape && absl::ascii_isxdigit(c)))) {
          dest.append("\\" "x");
          dest.push_back(hex_char[c / 16]);
          dest.push_back(hex_char[c % 16]);
          is_hex_escape = true;
        } else {
          dest.push_back(c);
          break;
        }
    }
    last_hex_escape = is_hex_escape;
  }

  return dest;
}

std::string ToStringLiteral(absl::string_view str) {
  absl::string_view quote =
      (str.find('"') != str.npos && str.find('\'') == str.npos) ? "'" : "\"";
  return absl::StrCat(
      quote, CEscapeInternal(str, true /* utf8_safe */, quote[0]), quote);
}

}  // namespace bigquery_ml_utils
