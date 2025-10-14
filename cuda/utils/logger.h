/**
 * @file logger.h
 * @brief Enterprise-grade logging system for CUDA matrix multiplication benchmarks
 * @author Jesse Moses (@Cre4T3Tiv3)
 * @date 2025
 * @copyright ByteStack Labs - MIT License
 *
 * Provides thread-safe, configurable logging with multiple output destinations
 * and severity levels for high-performance computing applications.
 */

#ifndef LOGGER_H
#define LOGGER_H

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

/**
 * @brief Log severity levels following standard practice
 */
enum class LogLevel {
  DEBUG = 0,    ///< Detailed information for debugging
  INFO = 1,     ///< General information about program execution
  WARNING = 2,  ///< Warning messages for potential issues
  ERROR = 3,    ///< Error messages for recoverable failures
  CRITICAL = 4  ///< Critical errors that may cause program termination
};

/**
 * @brief Thread-safe singleton logger class
 *
 * Provides enterprise-grade logging functionality with multiple output
 * destinations, configurable log levels, and automatic timestamping.
 * Designed for high-performance computing applications where reliable
 * logging is essential for debugging and performance analysis.
 */
class Logger {
 public:
  /**
   * @brief Get the singleton logger instance
   * @return Reference to the global logger instance
   */
  static Logger& getInstance() {
    static Logger instance;
    return instance;
  }

  /**
   * @brief Ensure log file is writable by attempting to remove root-owned files
   * @param log_file_path Path to the log file
   * @return true if file is writable, false otherwise
   */
  bool ensureLogFileWritable(const std::string& log_file_path) {
    // Try to open file for writing (will fail if permission denied)
    std::ofstream test_file(log_file_path, std::ios::app);
    if (test_file.is_open()) {
      test_file.close();
      return true;
    }

    // File exists but can't be opened - try to remove it
    if (std::remove(log_file_path.c_str()) == 0) {
      std::cerr << "Removed root-owned log file: " << log_file_path << std::endl;
      return true;
    }

    // Can't remove it either - permission denied
    std::cerr << "ERROR: Cannot write to or remove log file: " << log_file_path << std::endl;
    std::cerr << "This file was created by a previous run with sudo." << std::endl;
    std::cerr << "Fix: Run 'sudo chown $USER:$USER " << log_file_path << "' or 'sudo rm "
              << log_file_path << "'" << std::endl;
    return false;
  }

  /**
   * @brief Initialize logger with file output
   * @param log_file_path Path to the log file
   * @param min_level Minimum log level to output
   * @param console_output Enable/disable console output
   * @return true if initialization successful, false otherwise
   */
  bool initialize(const std::string& log_file_path, LogLevel min_level = LogLevel::INFO,
                  bool console_output = true) {
    std::lock_guard<std::mutex> lock(mutex_);

    min_level_ = min_level;
    console_output_ = console_output;

    if (!log_file_path.empty()) {
      // Ensure log file is writable before attempting to open
      if (!ensureLogFileWritable(log_file_path)) {
        std::cerr << "WARNING: Continuing with console-only logging" << std::endl;
        file_output_ = false;
      } else {
        log_file_.open(log_file_path, std::ios::app);
        if (!log_file_.is_open()) {
          std::cerr << "WARNING: Failed to open log file: " << log_file_path << std::endl;
          std::cerr << "WARNING: Continuing with console-only logging" << std::endl;
          file_output_ = false;
        } else {
          file_output_ = true;

          // Write initialization marker
          log_file_ << "\n" << std::string(80, '=') << "\n";
          log_file_ << "Logger initialized: " << getCurrentTimestamp() << "\n";
          log_file_ << std::string(80, '=') << "\n";
        }
      }
    } else {
      file_output_ = false;
    }

    initialized_ = true;
    return true;
  }

  /**
   * @brief Log a message with specified severity level
   * @param level Severity level of the message
   * @param file Source file name (__FILE__)
   * @param line Source line number (__LINE__)
   * @param function Function name (__FUNCTION__)
   * @param message Log message
   */
  void log(LogLevel level, const char* file, int line, const char* function,
           const std::string& message) {
    if (!initialized_ || level < min_level_) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    std::string formatted_message = formatMessage(level, file, line, function, message);

    if (console_output_) {
      if (level >= LogLevel::ERROR) {
        std::cerr << formatted_message << std::endl;
      } else {
        std::cout << formatted_message << std::endl;
      }
    }

    if (file_output_ && log_file_.is_open()) {
      log_file_ << formatted_message << std::endl;
      log_file_.flush();  // Ensure immediate write for critical applications
    }
  }

  /**
   * @brief Log to file only with full metadata, output clean message to console
   * @param level Severity level of the message
   * @param file Source file name (__FILE__)
   * @param line Source line number (__LINE__)
   * @param function Function name (__FUNCTION__)
   * @param console_message Clean message for console output
   * @param file_message Optional detailed message for file (uses console_message if empty)
   */
  void logCleanConsole(LogLevel level, const char* file, int line, const char* function,
                       const std::string& console_message, const std::string& file_message = "") {
    if (!initialized_ || level < min_level_) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Clean output to console
    if (console_output_) {
      if (level >= LogLevel::ERROR) {
        std::cerr << console_message << std::endl;
      } else {
        std::cout << console_message << std::endl;
      }
    }

    // Full metadata to file
    if (file_output_ && log_file_.is_open()) {
      std::string msg_for_file = file_message.empty() ? console_message : file_message;
      std::string formatted_message = formatMessage(level, file, line, function, msg_for_file);
      log_file_ << formatted_message << std::endl;
      log_file_.flush();
    }
  }

  /**
   * @brief Output message to console only (no file logging)
   * @param message Message to output to console
   * @param use_stderr If true, output to stderr instead of stdout
   */
  void consoleOnly(const std::string& message, bool use_stderr = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (console_output_) {
      if (use_stderr) {
        std::cerr << message << std::endl;
      } else {
        std::cout << message << std::endl;
      }
    }
  }

  /**
   * @brief Set minimum log level
   * @param level New minimum log level
   */
  void setLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    min_level_ = level;
  }

  /**
   * @brief Enable or disable console output
   * @param enable true to enable console output, false to disable
   */
  void setConsoleOutput(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    console_output_ = enable;
  }

  /**
   * @brief Cleanup and close log file
   */
  void shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_output_ && log_file_.is_open()) {
      log_file_ << "Logger shutdown: " << getCurrentTimestamp() << "\n";
      log_file_ << std::string(80, '=') << "\n";
      log_file_.close();
    }
    initialized_ = false;
  }

  // Delete copy constructor and assignment operator
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

 private:
  Logger() = default;
  ~Logger() { shutdown(); }

  /**
   * @brief Format log message with timestamp and metadata
   */
  std::string formatMessage(LogLevel level, const char* file, int line, const char* function,
                            const std::string& message) {
    std::ostringstream oss;

    oss << "[" << getCurrentTimestamp() << "] ";
    oss << "[" << levelToString(level) << "] ";

    // Extract filename from full path
    const char* filename = strrchr(file, '/');
    filename = filename ? filename + 1 : file;

    oss << "[" << filename << ":" << line << ":" << function << "] ";
    oss << message;

    return oss.str();
  }

  /**
   * @brief Convert log level to string representation
   */
  std::string levelToString(LogLevel level) {
    switch (level) {
      case LogLevel::DEBUG:
        return "DEBUG";
      case LogLevel::INFO:
        return "INFO ";
      case LogLevel::WARNING:
        return "WARN ";
      case LogLevel::ERROR:
        return "ERROR";
      case LogLevel::CRITICAL:
        return "CRIT ";
      default:
        return "UNKN ";
    }
  }

  /**
   * @brief Get current timestamp as formatted string
   */
  std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
  }

  std::mutex mutex_;
  std::ofstream log_file_;
  LogLevel min_level_ = LogLevel::INFO;
  bool console_output_ = true;
  bool file_output_ = false;
  bool initialized_ = false;
};

// ===============================
// ORIGINAL LOGGING MACROS (with full metadata)
// ===============================

// Basic logging macros (non-formatted)
#define LOG_DEBUG(msg) \
  Logger::getInstance().log(LogLevel::DEBUG, __FILE__, __LINE__, __FUNCTION__, msg)

#define LOG_INFO(msg) \
  Logger::getInstance().log(LogLevel::INFO, __FILE__, __LINE__, __FUNCTION__, msg)

#define LOG_WARNING(msg) \
  Logger::getInstance().log(LogLevel::WARNING, __FILE__, __LINE__, __FUNCTION__, msg)

#define LOG_ERROR(msg) \
  Logger::getInstance().log(LogLevel::ERROR, __FILE__, __LINE__, __FUNCTION__, msg)

#define LOG_CRITICAL(msg) \
  Logger::getInstance().log(LogLevel::CRITICAL, __FILE__, __LINE__, __FUNCTION__, msg)

// Formatted logging macros (printf-style)
#define LOG_DEBUG_F(fmt, ...)                           \
  do {                                                  \
    char buffer[1024];                                  \
    snprintf(buffer, sizeof(buffer), fmt, __VA_ARGS__); \
    LOG_DEBUG(std::string(buffer));                     \
  } while (0)

#define LOG_INFO_F(fmt, ...)                            \
  do {                                                  \
    char buffer[1024];                                  \
    snprintf(buffer, sizeof(buffer), fmt, __VA_ARGS__); \
    LOG_INFO(std::string(buffer));                      \
  } while (0)

#define LOG_WARNING_F(fmt, ...)                         \
  do {                                                  \
    char buffer[1024];                                  \
    snprintf(buffer, sizeof(buffer), fmt, __VA_ARGS__); \
    LOG_WARNING(std::string(buffer));                   \
  } while (0)

#define LOG_ERROR_F(fmt, ...)                           \
  do {                                                  \
    char buffer[1024];                                  \
    snprintf(buffer, sizeof(buffer), fmt, __VA_ARGS__); \
    LOG_ERROR(std::string(buffer));                     \
  } while (0)

#define LOG_CRITICAL_F(fmt, ...)                        \
  do {                                                  \
    char buffer[1024];                                  \
    snprintf(buffer, sizeof(buffer), fmt, __VA_ARGS__); \
    LOG_CRITICAL(std::string(buffer));                  \
  } while (0)

// ===============================
// CLEAN CONSOLE OUTPUT MACROS
// ===============================
// Clean console output with file logging (includes metadata in file)
#define PRINT_DEBUG(console_msg)                                                           \
  Logger::getInstance().logCleanConsole(LogLevel::DEBUG, __FILE__, __LINE__, __FUNCTION__, \
                                        console_msg)

#define PRINT_INFO(console_msg)                                                           \
  Logger::getInstance().logCleanConsole(LogLevel::INFO, __FILE__, __LINE__, __FUNCTION__, \
                                        console_msg)

#define PRINT_WARNING(console_msg)                                                           \
  Logger::getInstance().logCleanConsole(LogLevel::WARNING, __FILE__, __LINE__, __FUNCTION__, \
                                        console_msg)

#define PRINT_ERROR(console_msg)                                                           \
  Logger::getInstance().logCleanConsole(LogLevel::ERROR, __FILE__, __LINE__, __FUNCTION__, \
                                        console_msg)

#define PRINT_CRITICAL(console_msg)                                                           \
  Logger::getInstance().logCleanConsole(LogLevel::CRITICAL, __FILE__, __LINE__, __FUNCTION__, \
                                        console_msg)

// Clean console output with different file message
#define PRINT_DEBUG_DETAILED(console_msg, file_msg)                                        \
  Logger::getInstance().logCleanConsole(LogLevel::DEBUG, __FILE__, __LINE__, __FUNCTION__, \
                                        console_msg, file_msg)

#define PRINT_INFO_DETAILED(console_msg, file_msg)                                        \
  Logger::getInstance().logCleanConsole(LogLevel::INFO, __FILE__, __LINE__, __FUNCTION__, \
                                        console_msg, file_msg)

#define PRINT_WARNING_DETAILED(console_msg, file_msg)                                        \
  Logger::getInstance().logCleanConsole(LogLevel::WARNING, __FILE__, __LINE__, __FUNCTION__, \
                                        console_msg, file_msg)

#define PRINT_ERROR_DETAILED(console_msg, file_msg)                                        \
  Logger::getInstance().logCleanConsole(LogLevel::ERROR, __FILE__, __LINE__, __FUNCTION__, \
                                        console_msg, file_msg)

#define PRINT_CRITICAL_DETAILED(console_msg, file_msg)                                        \
  Logger::getInstance().logCleanConsole(LogLevel::CRITICAL, __FILE__, __LINE__, __FUNCTION__, \
                                        console_msg, file_msg)

// Formatted clean console output
#define PRINT_INFO_F(fmt, ...)                          \
  do {                                                  \
    char buffer[1024];                                  \
    snprintf(buffer, sizeof(buffer), fmt, __VA_ARGS__); \
    PRINT_INFO(std::string(buffer));                    \
  } while (0)

#define PRINT_WARNING_F(fmt, ...)                       \
  do {                                                  \
    char buffer[1024];                                  \
    snprintf(buffer, sizeof(buffer), fmt, __VA_ARGS__); \
    PRINT_WARNING(std::string(buffer));                 \
  } while (0)

#define PRINT_ERROR_F(fmt, ...)                         \
  do {                                                  \
    char buffer[1024];                                  \
    snprintf(buffer, sizeof(buffer), fmt, __VA_ARGS__); \
    PRINT_ERROR(std::string(buffer));                   \
  } while (0)

// ===============================
// CONSOLE-ONLY MACROS (no file logging)
// ===============================

#define CONSOLE_PRINT(msg) Logger::getInstance().consoleOnly(msg, false)

#define CONSOLE_ERROR(msg) Logger::getInstance().consoleOnly(msg, true)

#define CONSOLE_PRINT_F(fmt, ...)                       \
  do {                                                  \
    char buffer[1024];                                  \
    snprintf(buffer, sizeof(buffer), fmt, __VA_ARGS__); \
    CONSOLE_PRINT(std::string(buffer));                 \
  } while (0)

#define CONSOLE_ERROR_F(fmt, ...)                       \
  do {                                                  \
    char buffer[1024];                                  \
    snprintf(buffer, sizeof(buffer), fmt, __VA_ARGS__); \
    CONSOLE_ERROR(std::string(buffer));                 \
  } while (0)

#endif  // LOGGER_H
