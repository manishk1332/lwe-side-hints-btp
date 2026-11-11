// src/cli/demo_runner.cpp
//
// Paper Reference: Dachman-Soled et al. (CRYPTO 2020)
// Mathematical Purpose: Step-by-step execution engine for the DBDD interactive demo.
// Serializes intermediate lattice bases, ellipsoids, volumes, and secret recovery states
// into structured JSON for frontend presentation and slide decks.

#include "dbdd/dbdd/dbdd_instance.hpp"
#include "dbdd/dbdd/homogenization.hpp"
#include "dbdd/dbdd/isotropization.hpp"
#include "dbdd/hints/approximate_hint.hpp"
#include "dbdd/hints/modular_hint.hpp"
#include "dbdd/hints/perfect_hint.hpp"
#include "dbdd/hints/short_vector_hint.hpp"
#include "dbdd/lwe/distribution.hpp"
#include "dbdd/lwe/embedding.hpp"
#include "dbdd/lwe/lwe_instance.hpp"
#include "dbdd/lwe/secret_recovery.hpp"
#include "dbdd/reduction/bkz.hpp"
#include "dbdd/reduction/lll_exact.hpp"
#include "dbdd/security/security_estimator.hpp"

#include <chrono>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace dbdd;

// =============================================================================
// Lightweight JSON Parser and Serializer
// =============================================================================

struct JsonVal {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    std::vector<JsonVal> arr_val;
    std::map<std::string, JsonVal> obj_val;

    bool is_null() const { return type == Null; }
    bool is_bool() const { return type == Bool; }
    bool is_number() const { return type == Number; }
    bool is_string() const { return type == String; }
    bool is_array() const { return type == Array; }
    bool is_object() const { return type == Object; }

    bool get_bool(bool def = false) const { return is_bool() ? bool_val : def; }
    double get_number(double def = 0.0) const { return is_number() ? num_val : def; }
    std::string get_string(const std::string& def = "") const { return is_string() ? str_val : def; }
    const std::vector<JsonVal>& get_array() const { return arr_val; }
    const std::map<std::string, JsonVal>& get_object() const { return obj_val; }

    bool has_key(const std::string& k) const {
        return is_object() && obj_val.find(k) != obj_val.end();
    }

    const JsonVal& operator[](const std::string& k) const {
        static JsonVal null_v;
        if (!is_object()) return null_v;
        auto it = obj_val.find(k);
        return (it != obj_val.end()) ? it->second : null_v;
    }

    const JsonVal& operator[](size_t idx) const {
        static JsonVal null_v;
        if (!is_array() || idx >= arr_val.size()) return null_v;
        return arr_val[idx];
    }
};

class JsonParser {
public:
    static JsonVal parse(const std::string& src) {
        size_t pos = 0;
        skip_ws(src, pos);
        return parse_value(src, pos);
    }

private:
    static void skip_ws(const std::string& s, size_t& pos) {
        while (pos < s.size() && (std::isspace(static_cast<unsigned char>(s[pos])) || s[pos] == '\r' || s[pos] == '\n')) {
            ++pos;
        }
    }

    static JsonVal parse_value(const std::string& s, size_t& pos) {
        skip_ws(s, pos);
        if (pos >= s.size()) return JsonVal();

        char c = s[pos];
        if (c == '{') return parse_object(s, pos);
        if (c == '[') return parse_array(s, pos);
        if (c == '"') return parse_string(s, pos);
        if (c == 't' || c == 'f') return parse_bool(s, pos);
        if (c == 'n') return parse_null(s, pos);
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number(s, pos);

        return JsonVal();
    }

    static JsonVal parse_object(const std::string& s, size_t& pos) {
        JsonVal v;
        v.type = JsonVal::Object;
        ++pos; // skip '{'
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == '}') {
            ++pos;
            return v;
        }

        while (pos < s.size()) {
            skip_ws(s, pos);
            if (pos >= s.size() || s[pos] != '"') break;
            JsonVal k = parse_string(s, pos);
            skip_ws(s, pos);
            if (pos >= s.size() || s[pos] != ':') break;
            ++pos; // skip ':'
            JsonVal val = parse_value(s, pos);
            v.obj_val[k.str_val] = val;
            skip_ws(s, pos);
            if (pos < s.size() && s[pos] == ',') {
                ++pos;
            } else if (pos < s.size() && s[pos] == '}') {
                ++pos;
                break;
            } else {
                break;
            }
        }
        return v;
    }

    static JsonVal parse_array(const std::string& s, size_t& pos) {
        JsonVal v;
        v.type = JsonVal::Array;
        ++pos; // skip '['
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ']') {
            ++pos;
            return v;
        }

        while (pos < s.size()) {
            JsonVal item = parse_value(s, pos);
            v.arr_val.push_back(item);
            skip_ws(s, pos);
            if (pos < s.size() && s[pos] == ',') {
                ++pos;
            } else if (pos < s.size() && s[pos] == ']') {
                ++pos;
                break;
            } else {
                break;
            }
        }
        return v;
    }

    static JsonVal parse_string(const std::string& s, size_t& pos) {
        JsonVal v;
        v.type = JsonVal::String;
        ++pos; // skip leading '"'
        std::string res;
        while (pos < s.size()) {
            char c = s[pos++];
            if (c == '"') {
                v.str_val = res;
                return v;
            }
            if (c == '\\' && pos < s.size()) {
                char esc = s[pos++];
                if (esc == '"') res.push_back('"');
                else if (esc == '\\') res.push_back('\\');
                else if (esc == '/') res.push_back('/');
                else if (esc == 'b') res.push_back('\b');
                else if (esc == 'f') res.push_back('\f');
                else if (esc == 'n') res.push_back('\n');
                else if (esc == 'r') res.push_back('\r');
                else if (esc == 't') res.push_back('\t');
                else res.push_back(esc);
            } else {
                res.push_back(c);
            }
        }
        v.str_val = res;
        return v;
    }

    static JsonVal parse_number(const std::string& s, size_t& pos) {
        JsonVal v;
        v.type = JsonVal::Number;
        size_t start = pos;
        if (pos < s.size() && s[pos] == '-') ++pos;
        while (pos < s.size() && (std::isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+' || s[pos] == '-')) {
            ++pos;
        }
        std::string num_str = s.substr(start, pos - start);
        try {
            v.num_val = std::stod(num_str);
        } catch (...) {
            v.num_val = 0.0;
        }
        return v;
    }

    static JsonVal parse_bool(const std::string& s, size_t& pos) {
        JsonVal v;
        v.type = JsonVal::Bool;
        if (s.compare(pos, 4, "true") == 0) {
            v.bool_val = true;
            pos += 4;
        } else if (s.compare(pos, 5, "false") == 0) {
            v.bool_val = false;
            pos += 5;
        }
        return v;
    }

    static JsonVal parse_null(const std::string& s, size_t& pos) {
        JsonVal v;
        v.type = JsonVal::Null;
        if (s.compare(pos, 4, "null") == 0) {
            pos += 4;
        }
        return v;
    }
};

static std::string escape_json(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        if (c == '"') o << "\\\"";
        else if (c == '\\') o << "\\\\";
        else if (c == '\b') o << "\\b";
        else if (c == '\f') o << "\\f";
        else if (c == '\n') o << "\\n";
        else if (c == '\r') o << "\\r";
        else if (c == '\t') o << "\\t";
        else if (static_cast<unsigned char>(c) <= 0x1f) {
            o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
        } else {
            o << c;
        }
    }
    return o.str();
}

static std::string matrix_zz_to_json(const MatrixZZ& mat) {
    std::ostringstream ss;
    ss << "[\n";
    for (size_t i = 0; i < mat.rows(); ++i) {
        ss << "      [";
        for (size_t j = 0; j < mat.cols(); ++j) {
            ss << "\"" << mat(i, j).to_string() << "\"" << (j + 1 < mat.cols() ? ", " : "");
        }
        ss << "]" << (i + 1 < mat.rows() ? ",\n" : "\n");
    }
    ss << "    ]";
    return ss.str();
}

static std::string vector_zz_to_json(const VectorZZ& vec) {
    std::ostringstream ss;
    ss << "[";
    for (size_t j = 0; j < vec.size(); ++j) {
        ss << "\"" << vec[j].to_string() << "\"" << (j + 1 < vec.size() ? ", " : "");
    }
    ss << "]";
    return ss.str();
}

static std::string matrix_double_to_json(const MatrixDouble& mat) {
    std::ostringstream ss;
    ss << "[\n";
    for (size_t i = 0; i < mat.rows(); ++i) {
        ss << "      [";
        for (size_t j = 0; j < mat.cols(); ++j) {
            ss << std::fixed << std::setprecision(4) << mat(i, j) << (j + 1 < mat.cols() ? ", " : "");
        }
        ss << "]" << (i + 1 < mat.rows() ? ",\n" : "\n");
    }
    ss << "    ]";
    return ss.str();
}

static std::string vector_double_to_json(const VectorDouble& vec) {
    std::ostringstream ss;
    ss << "[";
    for (size_t j = 0; j < vec.size(); ++j) {
        ss << std::fixed << std::setprecision(4) << vec[j] << (j + 1 < vec.size() ? ", " : "");
    }
    ss << "]";
    return ss.str();
}

// =============================================================================
// Pipeline Configuration & Step Logging Structures
// =============================================================================

struct HintSpec {
    std::string type; // "perfect", "modular", "approximate", "short_vector"
    std::string label;
    std::vector<double> vector_double;
    std::vector<long long> vector_int;
    double scalar_value = 0.0;
    long long modulus = 2;
    double variance = 0.25;
};

struct PipelineConfig {
    std::string preset_name = "custom";
    size_t n = 3;
    size_t m = 3;
    BigInt q = BigInt(1051);
    std::string dist = "ternary";
    double sigma = 1.0;
    uint64_t seed = 53;
    size_t bkz_blocksize = 0; // 0 = auto
    size_t bkz_tours = 15;
    std::vector<HintSpec> hints;
    // Optional manual secret override
    bool has_manual_secret = false;
    std::vector<long long> manual_secret;
};

// =============================================================================
// Curated Presets
// =============================================================================

static PipelineConfig get_preset(const std::string& name) {
    PipelineConfig cfg;
    cfg.preset_name = name;

    if (name == "toy_pipeline") {
        // Preset 1: Exact match with examples/01_toy_pipeline.cpp (n=3, m=3, q=1051)
        cfg.n = 3;
        cfg.m = 3;
        cfg.q = BigInt(1051);
        cfg.dist = "ternary";
        cfg.seed = 53;
        cfg.bkz_blocksize = 2;
        cfg.bkz_tours = 15;

        // Hint 1: Short-vector hint (q-vector projection)
        HintSpec h1;
        h1.type = "short_vector";
        h1.label = "Short-Vector Hint: Lattice q-vector projection";
        cfg.hints.push_back(h1);

        // Hint 2: Perfect hint s_0 = 1
        HintSpec h2;
        h2.type = "perfect";
        h2.label = "Perfect Hint: Exact secret coordinate s_0 revealed";
        h2.vector_int = {1, 0, 0};
        cfg.hints.push_back(h2);

        // Hint 3: Approximate hint on s_1
        HintSpec h3;
        h3.type = "approximate";
        h3.label = "Approximate Hint: Noisy measurement s_1 + noise";
        h3.vector_double = {0.0, 1.0, 0.0};
        h3.variance = 0.50;
        cfg.hints.push_back(h3);

    } else if (name == "underdetermined") {
        // Preset 2: Underdetermined LWE (n=5, m=3, q=257) - m < n information deficit
        cfg.n = 5;
        cfg.m = 3;
        cfg.q = BigInt(257);
        cfg.dist = "ternary";
        cfg.seed = 101;
        cfg.bkz_blocksize = 2;
        cfg.bkz_tours = 15;

        // Hint 1: Perfect hint s_0
        HintSpec h1;
        h1.type = "perfect";
        h1.label = "Perfect Hint: Coordinate s_0";
        h1.vector_int = {1, 0, 0, 0, 0};
        cfg.hints.push_back(h1);

        // Hint 2: Perfect hint s_1
        HintSpec h2;
        h2.type = "perfect";
        h2.label = "Perfect Hint: Coordinate s_1";
        h2.vector_int = {0, 1, 0, 0, 0};
        cfg.hints.push_back(h2);

        // Hint 3: Modular hint s_2 (mod 2)
        HintSpec h3;
        h3.type = "modular";
        h3.label = "Modular Hint: Coordinate s_2 (mod 2)";
        h3.vector_int = {0, 0, 1, 0, 0};
        h3.modulus = 2;
        cfg.hints.push_back(h3);

    } else if (name == "overdetermined") {
        // Preset 3: Overdetermined LWE (n=3, m=6, q=257) - m > n with discrete Gaussian noise
        cfg.n = 3;
        cfg.m = 6;
        cfg.q = BigInt(257);
        cfg.dist = "gaussian";
        cfg.sigma = 1.2;
        cfg.seed = 202;
        cfg.bkz_blocksize = 2;
        cfg.bkz_tours = 15;

        // Hint 1: Modular hint s_0 (mod 3)
        HintSpec h1;
        h1.type = "modular";
        h1.label = "Modular Hint: Secret coordinate s_0 (mod 3)";
        h1.vector_int = {1, 0, 0};
        h1.modulus = 3;
        cfg.hints.push_back(h1);

        // Hint 2: Modular hint s_1 (mod 3)
        HintSpec h2;
        h2.type = "modular";
        h2.label = "Modular Hint: Secret coordinate s_1 (mod 3)";
        h2.vector_int = {0, 1, 0};
        h2.modulus = 3;
        cfg.hints.push_back(h2);

        // Hint 3: Approximate hint on s_2
        HintSpec h3;
        h3.type = "approximate";
        h3.label = "Approximate Hint: Secret coordinate s_2 with noise";
        h3.vector_double = {0.0, 0.0, 1.0};
        h3.variance = 0.35;
        cfg.hints.push_back(h3);

    } else if (name == "structural") {
        // Preset 4: Structural Balanced Secret (n=6, m=4, q=521) - sum(s_i) = 0
        cfg.n = 6;
        cfg.m = 4;
        cfg.q = BigInt(521);
        cfg.dist = "ternary";
        cfg.seed = 303;
        cfg.bkz_blocksize = 2;
        cfg.bkz_tours = 15;
        cfg.has_manual_secret = true;
        cfg.manual_secret = {1, 1, -1, -1, 0, 0}; // balanced sum = 0

        // Hint 1: Structural balance hint sum(s_i) = 0
        HintSpec h1;
        h1.type = "perfect";
        h1.label = "Structural Hint: Balanced Secret sum(s_i) = 0";
        h1.vector_int = {1, 1, 1, 1, 1, 1};
        h1.scalar_value = 0.0;
        cfg.hints.push_back(h1);

        // Hint 2: Difference measurement s_0 - s_1
        HintSpec h2;
        h2.type = "approximate";
        h2.label = "Approximate Hint: Leakage on s_0 - s_1";
        h2.vector_double = {1.0, -1.0, 0.0, 0.0, 0.0, 0.0};
        h2.variance = 0.30;
        cfg.hints.push_back(h2);

    } else if (name == "higher_dim") {
        // Preset 5: Higher-Dimension Challenge (n=10, m=12, q=1051)
        cfg.n = 10;
        cfg.m = 12;
        cfg.q = BigInt(1051);
        cfg.dist = "ternary";
        cfg.seed = 404;
        cfg.bkz_blocksize = 3;
        cfg.bkz_tours = 15;

        for (size_t i = 0; i < 3; ++i) {
            HintSpec h;
            h.type = "perfect";
            h.label = "Perfect Hint: Coordinate s_" + std::to_string(i);
            h.vector_int = std::vector<long long>(cfg.n, 0);
            h.vector_int[i] = 1;
            cfg.hints.push_back(h);
        }

        for (size_t i = 3; i < 5; ++i) {
            HintSpec h;
            h.type = "approximate";
            h.label = "Approximate Hint: Coordinate s_" + std::to_string(i);
            h.vector_double = std::vector<double>(cfg.n, 0.0);
            h.vector_double[i] = 1.0;
            h.variance = 0.25;
            cfg.hints.push_back(h);
        }
    }
    return cfg;
}

// =============================================================================
// JSON Config Parser
// =============================================================================

static PipelineConfig parse_pipeline_config(const JsonVal& root) {
    PipelineConfig cfg;
    if (root.has_key("preset") && !root["preset"].get_string().empty()) {
        std::string preset_name = root["preset"].get_string();
        if (preset_name != "custom") {
            cfg = get_preset(preset_name);
            return cfg;
        }
    }

    cfg.preset_name = root.has_key("preset_name") ? root["preset_name"].get_string() : "custom";
    cfg.n = static_cast<size_t>(root["n"].get_number(3));
    cfg.m = static_cast<size_t>(root["m"].get_number(3));
    if (root.has_key("q")) {
        if (root["q"].is_string()) {
            cfg.q = BigInt(root["q"].get_string());
        } else {
            cfg.q = BigInt(static_cast<long long>(root["q"].get_number(1051)));
        }
    }
    cfg.dist = root["dist"].get_string("ternary");
    cfg.sigma = root["sigma"].get_number(1.0);
    cfg.seed = static_cast<uint64_t>(root["seed"].get_number(53));
    cfg.bkz_blocksize = static_cast<size_t>(root["bkz_blocksize"].get_number(0));
    cfg.bkz_tours = static_cast<size_t>(root["bkz_tours"].get_number(15));

    if (root.has_key("hints") && root["hints"].is_array()) {
        for (const auto& h_val : root["hints"].get_array()) {
            HintSpec h;
            h.type = h_val["type"].get_string("perfect");
            h.label = h_val["label"].get_string();
            h.scalar_value = h_val["scalar_value"].get_number(0.0);
            h.modulus = static_cast<long long>(h_val["modulus"].get_number(2));
            h.variance = h_val["variance"].get_number(0.25);

            if (h_val.has_key("vector") && h_val["vector"].is_array()) {
                for (const auto& elem : h_val["vector"].get_array()) {
                    double val = elem.get_number(0.0);
                    h.vector_double.push_back(val);
                    h.vector_int.push_back(static_cast<long long>(std::round(val)));
                }
            }
            cfg.hints.push_back(h);
        }
    }

    return cfg;
}

// =============================================================================
// Pipeline Runner
// =============================================================================

static int execute_pipeline(const PipelineConfig& cfg, std::ostream& out) {
    auto start_all = std::chrono::high_resolution_clock::now();

    // 1. LWE Instance Generation
    std::unique_ptr<Distribution> s_dist;
    std::unique_ptr<Distribution> e_dist;

    if (cfg.dist == "gaussian") {
        s_dist = std::make_unique<DiscreteGaussian>(cfg.sigma, cfg.seed);
        e_dist = std::make_unique<DiscreteGaussian>(cfg.sigma, cfg.seed + 1);
    } else if (cfg.dist == "binomial") {
        size_t k = static_cast<size_t>(std::max(1.0, std::round(2.0 * cfg.sigma * cfg.sigma)));
        s_dist = std::make_unique<CenteredBinomial>(k, cfg.seed);
        e_dist = std::make_unique<CenteredBinomial>(k, cfg.seed + 1);
    } else {
        s_dist = std::make_unique<UniformTernary>(2.0 / 3.0, cfg.seed);
        e_dist = std::make_unique<UniformTernary>(2.0 / 3.0, cfg.seed + 1);
    }

    LWEInstance lwe = LWEInstance::generate_random(cfg.n, cfg.m, cfg.q, *s_dist, *e_dist, cfg.seed);

    // If manual balanced secret is provided (e.g. for structural preset)
    if (cfg.has_manual_secret && cfg.manual_secret.size() == cfg.n) {
        VectorZZ secret(cfg.n);
        for (size_t i = 0; i < cfg.n; ++i) secret[i] = BigInt(cfg.manual_secret[i]);
        VectorZZ sA = secret * lwe.A().transpose();
        VectorZZ b(cfg.m);
        for (size_t i = 0; i < cfg.m; ++i) {
            BigInt val = (sA[i] + lwe.test_only_error()[i]) % cfg.q;
            if (val < BigInt(0)) val += cfg.q;
            b[i] = val;
        }
        lwe = LWEInstance(cfg.n, cfg.m, cfg.q, lwe.A(), b, secret, lwe.test_only_error());
    }

    SecretReconstructor reconstructor(lwe);
    auto kannan = KannanEmbedding::from_lwe(lwe, *e_dist, *s_dist);
    DBDDInstance dbdd = DBDDInstance::from_kannan(kannan);

    std::vector<std::string> step_jsons;
    size_t step_idx = 1;

    size_t prev_dim = dbdd.dimension();
    double prev_vol = dbdd.volume().to_double();

    // Helper lambda to format a step JSON
    auto record_step = [&](const std::string& name,
                           const std::string& stage,
                           const std::string& description,
                           const std::string& extra_fields = "") {
        size_t cur_dim = dbdd.dimension();
        double cur_vol = dbdd.volume().to_double();
        long long delta_dim = static_cast<long long>(cur_dim) - static_cast<long long>(prev_dim);
        double delta_vol_ratio = (prev_vol > 0.0) ? (cur_vol / prev_vol) : 1.0;
        prev_dim = cur_dim;
        prev_vol = cur_vol;

        std::ostringstream s;
        s << "    {\n"
          << "      \"step_index\": " << step_idx++ << ",\n"
          << "      \"step_name\": \"" << escape_json(name) << "\",\n"
          << "      \"stage\": \"" << stage << "\",\n"
          << "      \"description\": \"" << escape_json(description) << "\",\n"
          << "      \"metrics\": {\n"
          << "        \"dimension\": " << cur_dim << ",\n"
          << "        \"ambient_dimension\": " << dbdd.ambient_dimension() << ",\n"
          << "        \"volume\": " << cur_vol << ",\n"
          << "        \"log2_volume\": " << (dbdd.log_volume().to_double() / std::log(2.0)) << ",\n"
          << "        \"covariance_rank\": " << dbdd.ellipsoid().rank() << ",\n"
          << "        \"target_in_lattice\": " << (dbdd.has_target() && dbdd.target_in_lattice() ? "true" : "false") << ",\n"
          << "        \"delta_dim\": " << delta_dim << ",\n"
          << "        \"delta_vol_ratio\": " << delta_vol_ratio << "\n"
          << "      },\n"
          << "      \"basis\": " << matrix_zz_to_json(dbdd.matrix()) << ",\n"
          << "      \"mean\": " << vector_double_to_json(dbdd.mean()) << ",\n"
          << "      \"covariance\": " << matrix_double_to_json(dbdd.covariance()) << ",\n"
          << "      \"target\": " << (dbdd.has_target() ? vector_zz_to_json(dbdd.target()) : "[]")
          << (extra_fields.empty() ? "" : ",\n" + extra_fields) << "\n"
          << "    }";
        step_jsons.push_back(s.str());
    };

    // Step 1: Kannan Embedding (Initial DBDD)
    auto gsa_init = SecurityEstimator::gsa_intersect(dbdd);
    auto prob_init = SecurityEstimator::probabilistic(dbdd);
    std::ostringstream est_extra;
    est_extra << "      \"estimator\": {\n"
              << "        \"beta_gsa\": " << gsa_init.beta << ",\n"
              << "        \"beta_prob\": " << prob_init.beta << ",\n"
              << "        \"delta\": " << std::fixed << std::setprecision(4) << gsa_init.delta() << "\n"
              << "      }";

    record_step(
        "Kannan Embedding -> Initial DBDD Instance",
        "kannan",
        "LWE system embedded into an (m + n + 1)-dimensional Kannan lattice B with target vector s_bar = (e, s, 1) and diagonal covariance prior Sigma.",
        est_extra.str());

    // Steps 2..k: Side Hint Integrations
    for (size_t hi = 0; hi < cfg.hints.size(); ++hi) {
        const auto& h = cfg.hints[hi];
        std::string h_name = "Hint " + std::to_string(hi + 1) + ": ";

        if (h.type == "short_vector") {
            VectorZZ q_vec = dbdd.basis().row(0);
            reconstructor.record_projection(q_vec);
            dbdd.integrate_short_vector_hint(q_vec);
            record_step(
                h_name + (h.label.empty() ? "Short-Vector Projection" : h.label),
                "hint",
                "Lattice projected orthogonal to short q-vector. Dimension drops by 1 and volume contracts by ||v||.");
        } else if (h.type == "perfect") {
            VectorZZ v(cfg.n, BigInt(0));
            BigInt l(0);
            if (!h.vector_int.empty()) {
                size_t sz = std::min(cfg.n, h.vector_int.size());
                for (size_t j = 0; j < sz; ++j) v[j] = BigInt(h.vector_int[j]);
                // Compute true scalar <s, v> if not manually set
                for (size_t j = 0; j < cfg.n; ++j) {
                    l += lwe.test_only_secret()[j] * v[j];
                }
            } else {
                v[0] = BigInt(1);
                l = lwe.test_only_secret()[0];
            }
            dbdd.integrate_perfect_hint(v, l);
            record_step(
                h_name + (h.label.empty() ? "Perfect Hint Integration" : h.label),
                "hint",
                "Exact linear hyperplane constraint <s, v> = l integrated via dual projection. Primal dimension drops by 1.");
        } else if (h.type == "modular") {
            VectorZZ v(cfg.n, BigInt(0));
            BigInt k(h.modulus > 1 ? h.modulus : 2);
            if (!h.vector_int.empty()) {
                size_t sz = std::min(cfg.n, h.vector_int.size());
                for (size_t j = 0; j < sz; ++j) v[j] = BigInt(h.vector_int[j]);
            } else {
                v[0] = BigInt(1);
            }
            BigInt prod(0);
            for (size_t j = 0; j < cfg.n; ++j) prod += lwe.test_only_secret()[j] * v[j];
            BigInt l = prod % k;
            if (l < BigInt(0)) l += k;

            dbdd.integrate_modular_hint(v, l, k);
            record_step(
                h_name + (h.label.empty() ? "Modular Hint Integration" : h.label),
                "hint",
                "Periodic modular constraint <s, v> = l (mod " + k.to_string() + ") integrated. Primal lattice sparsified by factor k, preserving dimension.");
        } else if (h.type == "approximate") {
            VectorDouble v(cfg.n, 0.0);
            if (!h.vector_double.empty()) {
                size_t sz = std::min(cfg.n, h.vector_double.size());
                for (size_t j = 0; j < sz; ++j) v[j] = h.vector_double[j];
            } else {
                v[0] = 1.0;
            }
            double dot = 0.0;
            for (size_t j = 0; j < cfg.n; ++j) {
                dot += lwe.test_only_secret()[j].to_double() * v[j];
            }
            double noise_val = (h.variance > 0.0) ? std::sqrt(h.variance) * 0.3 : 0.0;
            double l_approx = dot + noise_val;
            double var = (h.variance > 0.0) ? h.variance : 0.25;

            dbdd.integrate_approximate_hint(v, l_approx, var);
            record_step(
                h_name + (h.label.empty() ? "Approximate Hint Integration" : h.label),
                "hint",
                "Gaussian side-channel measurement <s, v> ~ N(" + std::to_string(l_approx) + ", " + std::to_string(var) + ") updated via Bayes formula.");
        }
    }

    // Homogenization & Isotropization
    VectorDouble pre_mean = dbdd.mean();
    MatrixDouble pre_cov = dbdd.covariance();
    double pre_vol = dbdd.volume().to_double();

    dbdd.homogenize();
    MatrixDouble M_transform = Isotropization::compute_transform(dbdd.covariance());
    dbdd.isotropize();
    double post_vol = dbdd.volume().to_double();

    std::ostringstream homo_extra;
    homo_extra << "      \"homo_iso_details\": {\n"
               << "        \"pre_mean\": " << vector_double_to_json(pre_mean) << ",\n"
               << "        \"pre_covariance\": " << matrix_double_to_json(pre_cov) << ",\n"
               << "        \"transform_M\": " << matrix_double_to_json(M_transform) << ",\n"
               << "        \"pre_volume\": " << pre_vol << ",\n"
               << "        \"post_volume\": " << post_vol << ",\n"
               << "        \"volume_ratio\": " << (pre_vol > 0.0 ? (post_vol / pre_vol) : 1.0) << "\n"
               << "      }";

    record_step(
        "Homogenization & Isotropization",
        "transformation",
        "Distribution shifted to zero mean (mu = 0) and covariance sphere transformed to isotropic shape M^T Sigma M ~ I for optimal lattice reduction.",
        homo_extra.str());

    // Security Estimation
    auto gsa_final = SecurityEstimator::gsa_intersect(dbdd);
    auto prob_final = SecurityEstimator::probabilistic(dbdd);
    std::ostringstream est_post;
    est_post << "      \"estimator\": {\n"
             << "        \"beta_gsa\": " << gsa_final.beta << ",\n"
             << "        \"beta_continuous\": " << gsa_final.beta_continuous << ",\n"
             << "        \"delta\": " << std::fixed << std::setprecision(4) << gsa_final.delta() << ",\n"
             << "        \"beta_prob\": " << prob_final.beta << ",\n"
             << "        \"success_prob\": " << prob_final.success_probability << "\n"
             << "      }";

    record_step(
        "Post-Transformation Security Estimation",
        "estimation",
        "GSA-Intersect and Probabilistic Simulators predict required BKZ blocksize beta after hint degradation.",
        est_post.str());

    // BKZ Lattice Reduction
    size_t target_beta = (cfg.bkz_blocksize > 0) ? cfg.bkz_blocksize : std::max(size_t(2), gsa_final.beta);
    BKZParameters bkz_params(target_beta, cfg.bkz_tours, 0.99);

    // Compute norms before BKZ
    std::vector<double> before_norms;
    for (size_t r = 0; r < dbdd.matrix().rows(); ++r) {
        double sq = 0.0;
        for (size_t c = 0; c < dbdd.matrix().cols(); ++c) {
            double v = dbdd.matrix()(r, c).to_double();
            sq += v * v;
        }
        before_norms.push_back(std::sqrt(sq));
    }

    auto t_bkz0 = std::chrono::high_resolution_clock::now();
    auto bkz_res = bkz_reduce(dbdd.basis(), bkz_params);
    auto t_bkz1 = std::chrono::high_resolution_clock::now();
    double bkz_ms = std::chrono::duration<double, std::milli>(t_bkz1 - t_bkz0).count();

    // Compute norms after BKZ
    std::vector<double> after_norms;
    for (size_t r = 0; r < bkz_res.basis.dimension(); ++r) {
        double sq = 0.0;
        for (size_t c = 0; c < bkz_res.basis.matrix().cols(); ++c) {
            double v = bkz_res.basis.matrix()(r, c).to_double();
            sq += v * v;
        }
        after_norms.push_back(std::sqrt(sq));
    }

    // Secret Reconstruction & Verification
    const auto& red_basis = bkz_res.basis;
    bool recovered = false;
    VectorZZ final_secret(cfg.n);
    long long target_row_idx = -1;
    VectorZZ chosen_candidate;

    for (size_t r = 0; r < red_basis.dimension(); ++r) {
        const VectorZZ& candidate = red_basis.row(r);
        size_t amb = dbdd.ambient_dimension();
        if (candidate[amb - 1].abs() == BigInt(1)) {
            VectorZZ cand_s = reconstructor.recover_secret(candidate);
            if (reconstructor.verify_recovery(cand_s)) {
                recovered = true;
                final_secret = cand_s;
                target_row_idx = static_cast<long long>(r);
                chosen_candidate = candidate;
                break;
            }
        }
    }

    if (!recovered) {
        auto rec_dbdd = BKZReduction::recover_dbdd_secret(red_basis, dbdd);
        if (rec_dbdd.has_value()) {
            final_secret = reconstructor.recover_secret(*rec_dbdd);
            recovered = reconstructor.verify_recovery(final_secret);
            chosen_candidate = *rec_dbdd;
            target_row_idx = 0;
        }
    }

    // If still not matched, check if row 0 has secret
    if (!recovered && red_basis.dimension() > 0) {
        chosen_candidate = red_basis.row(0);
        target_row_idx = 0;
    }

    std::ostringstream bkz_extra;
    bkz_extra << "      \"bkz_info\": {\n"
              << "        \"blocksize\": " << target_beta << ",\n"
              << "        \"tours\": " << bkz_res.tours_completed << ",\n"
              << "        \"svp_calls\": " << bkz_res.total_svp_calls << ",\n"
              << "        \"runtime_ms\": " << bkz_ms << ",\n"
              << "        \"target_row_index\": " << target_row_idx << ",\n"
              << "        \"target_norm\": " << (target_row_idx >= 0 && target_row_idx < static_cast<long long>(after_norms.size()) ? after_norms[target_row_idx] : 0.0) << ",\n"
              << "        \"before_norms\": [";
    for (size_t i = 0; i < before_norms.size(); ++i) {
        bkz_extra << std::fixed << std::setprecision(2) << before_norms[i] << (i + 1 < before_norms.size() ? ", " : "");
    }
    bkz_extra << "],\n        \"after_norms\": [";
    for (size_t i = 0; i < after_norms.size(); ++i) {
        bkz_extra << std::fixed << std::setprecision(2) << after_norms[i] << (i + 1 < after_norms.size() ? ", " : "");
    }
    bkz_extra << "]\n      },\n"
              << "      \"reduced_basis\": " << matrix_zz_to_json(bkz_res.basis.matrix());

    record_step(
        "BKZ Lattice Reduction (beta=" + std::to_string(target_beta) + ")",
        "reduction",
        "Block Korkine-Zolotarev lattice reduction executed on transformed basis. Shortest vector in block discovered via exact SVP oracle.",
        bkz_extra.str());

    // Direct check if identical to ground truth
    bool exact_match = (recovered && (final_secret == lwe.test_only_secret()));
    if (!exact_match && final_secret == lwe.test_only_secret()) {
        exact_match = true;
        recovered = true;
    }

    // Reconstruction breakdown
    VectorZZ normalized_candidate = chosen_candidate;
    size_t amb_sz = dbdd.ambient_dimension();
    if (normalized_candidate.size() == amb_sz && normalized_candidate[amb_sz - 1] < BigInt(0)) {
        for (size_t j = 0; j < amb_sz; ++j) normalized_candidate[j] = -normalized_candidate[j];
    }

    VectorZZ full_target = reconstructor.recover_target(chosen_candidate);
    VectorZZ extracted_error(cfg.m);
    for (size_t i = 0; i < cfg.m && i < full_target.size(); ++i) {
        extracted_error[i] = full_target[i];
    }

    VectorZZ sA = final_secret * lwe.A().transpose();
    VectorZZ b_calc(cfg.m);
    for (size_t i = 0; i < cfg.m; ++i) {
        BigInt val = (sA[i] + extracted_error[i]) % cfg.q;
        if (val < BigInt(0)) val += cfg.q;
        b_calc[i] = val;
    }

    std::ostringstream rec_extra;
    rec_extra << "      \"recovery\": {\n"
              << "        \"ground_truth\": " << vector_zz_to_json(lwe.test_only_secret()) << ",\n"
              << "        \"recovered\": " << vector_zz_to_json(final_secret) << ",\n"
              << "        \"success\": " << (exact_match ? "true" : "false") << ",\n"
              << "        \"raw_candidate\": " << vector_zz_to_json(chosen_candidate) << ",\n"
              << "        \"normalized_candidate\": " << vector_zz_to_json(normalized_candidate) << ",\n"
              << "        \"extracted_secret\": " << vector_zz_to_json(final_secret) << ",\n"
              << "        \"extracted_error\": " << vector_zz_to_json(extracted_error) << ",\n"
              << "        \"calculated_b\": " << vector_zz_to_json(b_calc) << ",\n"
              << "        \"public_b\": " << vector_zz_to_json(lwe.b()) << "\n"
              << "      }";

    record_step(
        "Secret Reconstruction & Verification",
        "recovery",
        exact_match ? "Ground-truth secret s reconstructed and verified against public instance b = s A^T + e (mod q)."
                    : "Lattice reduced, but secret coordinates did not match ground truth.",
        rec_extra.str());

    auto end_all = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end_all - start_all).count();

    // Assemble Full Output JSON
    out << "{\n"
        << "  \"preset_name\": \"" << escape_json(cfg.preset_name) << "\",\n"
        << "  \"lwe\": {\n"
        << "    \"n\": " << cfg.n << ",\n"
        << "    \"m\": " << cfg.m << ",\n"
        << "    \"q\": \"" << cfg.q.to_string() << "\",\n"
        << "    \"dist\": \"" << cfg.dist << "\",\n"
        << "    \"sigma\": " << cfg.sigma << ",\n"
        << "    \"seed\": " << cfg.seed << ",\n"
        << "    \"matrix_A\": " << matrix_zz_to_json(lwe.A()) << ",\n"
        << "    \"vector_b\": " << vector_zz_to_json(lwe.b()) << ",\n"
        << "    \"ground_truth_secret\": " << vector_zz_to_json(lwe.test_only_secret()) << ",\n"
        << "    \"ground_truth_error\": " << vector_zz_to_json(lwe.test_only_error()) << "\n"
        << "  },\n"
        << "  \"steps\": [\n";

    for (size_t i = 0; i < step_jsons.size(); ++i) {
        out << step_jsons[i] << (i + 1 < step_jsons.size() ? ",\n" : "\n");
    }

    out << "  ],\n"
        << "  \"result\": {\n"
        << "    \"status\": \"" << (exact_match ? "SUCCESS" : "FAILURE") << "\",\n"
        << "    \"ground_truth_secret\": " << vector_zz_to_json(lwe.test_only_secret()) << ",\n"
        << "    \"recovered_secret\": " << vector_zz_to_json(final_secret) << ",\n"
        << "    \"exact_match\": " << (exact_match ? "true" : "false") << ",\n"
        << "    \"bkz_tours\": " << bkz_res.tours_completed << ",\n"
        << "    \"svp_calls\": " << bkz_res.total_svp_calls << ",\n"
        << "    \"total_runtime_ms\": " << total_ms << "\n"
        << "  }\n"
        << "}\n";

    return exact_match ? 0 : 1;
}

// =============================================================================
// CLI Entry Point
// =============================================================================

int main(int argc, char* argv[]) {
    try {
        std::string preset_arg = "";
        std::string json_input = "";
        bool test_mode = false;
        bool test_all = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--preset" && i + 1 < argc) {
                preset_arg = argv[++i];
            } else if (arg == "--test-preset" && i + 1 < argc) {
                preset_arg = argv[++i];
                test_mode = true;
            } else if (arg == "--test-all-presets") {
                test_all = true;
            } else if (arg == "--json" && i + 1 < argc) {
                json_input = argv[++i];
            } else if (arg == "--json-file" && i + 1 < argc) {
                std::ifstream f(argv[++i]);
                if (f) {
                    std::ostringstream ss;
                    ss << f.rdbuf();
                    json_input = ss.str();
                }
            } else if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: dbdd_demo_runner [options]\n"
                          << "  --preset <name>       Run named preset (toy_pipeline, underdetermined, overdetermined, structural, higher_dim)\n"
                          << "  --test-preset <name>  Run preset and assert success (exits 0 on match, 1 on fail)\n"
                          << "  --test-all-presets    Run all 5 presets and assert success\n"
                          << "  --json <str>          Input JSON payload (use '-' for stdin)\n"
                          << "  --json-file <path>    Input JSON file\n";
                return 0;
            }
        }

        if (test_all) {
            const std::vector<std::string> all_presets = {
                "toy_pipeline", "underdetermined", "overdetermined", "structural", "higher_dim"
            };
            int fails = 0;
            for (const auto& p : all_presets) {
                std::cout << "Testing preset: " << p << " ... ";
                std::ostringstream dummy;
                PipelineConfig cfg = get_preset(p);
                int rc = execute_pipeline(cfg, dummy);
                if (rc == 0) {
                    std::cout << "[PASS]\n";
                } else {
                    std::cout << "[FAIL]\n";
                    fails++;
                }
            }
            return fails == 0 ? 0 : 1;
        }

        PipelineConfig cfg;
        if (!preset_arg.empty()) {
            cfg = get_preset(preset_arg);
        } else if (!json_input.empty()) {
            if (json_input == "-") {
                std::ostringstream ss;
                ss << std::cin.rdbuf();
                json_input = ss.str();
            }
            JsonVal root = JsonParser::parse(json_input);
            cfg = parse_pipeline_config(root);
        } else {
            // Default: toy_pipeline preset
            cfg = get_preset("toy_pipeline");
        }

        if (test_mode) {
            std::ostringstream out;
            int rc = execute_pipeline(cfg, out);
            if (rc == 0) {
                std::cout << "Preset " << preset_arg << " passed verification!\n";
            } else {
                std::cerr << "Preset " << preset_arg << " failed secret recovery!\n";
            }
            return rc;
        }

        return execute_pipeline(cfg, std::cout);

    } catch (const DBDDException& ex) {
        std::cerr << "DBDD Exception: " << ex.what() << "\n";
        return 2;
    } catch (const std::exception& ex) {
        std::cerr << "Standard Exception: " << ex.what() << "\n";
        return 3;
    }
}

