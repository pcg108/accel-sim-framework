#include <trace_file_manager.h>

#ifdef TRACE_FILE_MANAGER_SECOND_TU
TraceFileManager *other_translation_unit_manager() {
  return &TraceFileManager::instance();
}
#else
#include <cassert>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
TraceFileManager *other_translation_unit_manager();
enum Fault { None, OpenDenied, OpenEmfileOnce, Write, FormattedWrite, Flush,
             Close, Buffer, Limit, InfiniteLimit };
static Fault fault = None;
static size_t flushes = 0;
extern "C" {
FILE *__real_fopen(const char *, const char *);
int __real_fputs(const char *, FILE *);
int __real_vfprintf(FILE *, const char *, va_list);
int __real___vfprintf_chk(FILE *, int, const char *, va_list);
int __real_fflush(FILE *);
int __real_fclose(FILE *);
int __real_setvbuf(FILE *, char *, int, size_t);
int __real_getrlimit(__rlimit_resource_t, struct rlimit *);
FILE *__wrap_fopen(const char *path, const char *mode) {
  if (fault == OpenDenied || fault == OpenEmfileOnce) {
    errno = fault == OpenDenied ? EACCES : EMFILE;
    if (fault == OpenEmfileOnce) fault = None;
    return nullptr;
  }
  return __real_fopen(path, mode);
}
int __wrap_fputs(const char *line, FILE *f) {
  if (fault == Write) { errno = ENOSPC; return EOF; }
  return __real_fputs(line, f);
}
int __wrap_vfprintf(FILE *f, const char *format, va_list args) {
  if (fault == FormattedWrite) { errno = ENOSPC; return -1; }
  return __real_vfprintf(f, format, args);
}
int __wrap___vfprintf_chk(FILE *f, int flag, const char *format, va_list args) {
  if (fault == FormattedWrite) { errno = ENOSPC; return -1; }
  return __real___vfprintf_chk(f, flag, format, args);
}
int __wrap_fflush(FILE *f) {
  ++flushes;
  if (fault == Flush && f != stderr) { errno = EIO; return EOF; }
  return __real_fflush(f);
}
int __wrap_fclose(FILE *f) {
  const int result = __real_fclose(f);
  if (fault == Close) { errno = EIO; return EOF; }
  return result;
}
int __wrap_setvbuf(FILE *f, char *b, int m, size_t n) {
  if (fault == Buffer) { errno = ENOMEM; return -1; }
  return __real_setvbuf(f, b, m, n);
}
int __wrap_getrlimit(__rlimit_resource_t resource, struct rlimit *limit) {
  if (fault == Limit) { errno = EIO; return -1; }
  if (fault == InfiniteLimit) {
    limit->rlim_cur = limit->rlim_max = RLIM_INFINITY;
    return 0;
  }
  return __real_getrlimit(resource, limit);
}
}

static std::string read(const fs::path &path) {
  std::ifstream input(path);
  assert(input.good());
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
static void seed(const fs::path &path, const std::string &contents) {
  std::ofstream out(path); out << contents;
}
static void capacity(const char *n) {
  assert(setenv("ACCELSIM_TRACE_MAX_OPEN_FILES", n, 1) == 0);
}
static void low_limit() {
  struct rlimit limit;
  assert(__real_getrlimit(RLIMIT_NOFILE, &limit) == 0);
  assert(limit.rlim_max >= 32);
  limit.rlim_cur = 32;
  assert(setrlimit(RLIMIT_NOFILE, &limit) == 0);
}

static void eviction(const fs::path &p) {
  capacity("2");
  seed(p / "a", "stale\n");
  auto &m = TraceFileManager::instance();
  m.writef((p / "a").string(), "prefix %u ", 7U);
  m.write_line((p / "b").string(), "B1\n");
  m.write_line((p / "a").string(), "body\n"); // A touched: B is the LRU.
  m.write_line((p / "c").string(), "C1\n");
  assert(m.statistics().evictions == 1);
  assert(read(p / "b") == "B1\n"); // Eviction flushed B.
  m.write_line((p / "a").string(), "A2\n");
  assert(m.statistics().reopens == 0); // Touching A really preserved it.
  m.write_line((p / "b").string(), "B2\n");
  assert(m.statistics().reopens == 1);
  assert(m.statistics().peak_open_files == 2);
  m.close_all();
  assert(read(p / "a") == "prefix 7 body\nA2\n");
  assert(read(p / "b") == "B1\nB2\n");
  assert(read(p / "c") == "C1\n");
  m.write_line((p / "a").string(), "late\n");
  m.close_all();
  assert(read(p / "a") == "prefix 7 body\nA2\nlate\n");
}
static void partial_writes(const fs::path &p) {
  capacity("1"); auto &m = TraceFileManager::instance();
  m.writef((p / "a").string(), "%s", "prefix ");
  m.write_line((p / "b").string(), "unrelated\n");
  m.writef((p / "a").string(), "%d", 42);
  m.write_line((p / "c").string(), "other\n");
  m.write_line((p / "a").string(), " trailer\n");
  m.close_all(); assert(read(p / "a") == "prefix 42 trailer\n");
}
static void kernel_close(const fs::path &p) {
  capacity("8"); auto &m = TraceFileManager::instance();
  const auto a=p/"kernel_7_named"/"x", b=p/"kernel_7"/"x",
             c=p/"kernel_70_named"/"x", d=p/"prefix_kernel_7_named"/"kernel_7_named"/"x";
  for (const auto &f : {a,b,c,d}) {
    fs::create_directories(f.parent_path()); m.write_line(f.string(), "first\n");
  }
  m.close_kernel_files(7, "named");
  assert(m.statistics().open_files == 1);
  for (const auto &f : {a,b,d}) assert(read(f) == "first\n");
  m.write_line(a.string(), "completion\n");
  m.close_all(); assert(read(a) == "first\ncompletion\n");
  assert(read(c) == "first\n");
}
static void low_capacity(const fs::path &p) {
  low_limit(); auto &m = TraceFileManager::instance();
  assert(m.statistics().capacity == 16);
  for (int pass=0;pass<3;++pass)
    for (int i=0;i<80;++i) m.writef((p/std::to_string(i)).string(), "%d\n", pass);
  assert(m.statistics().peak_open_files == 16);
  assert(m.statistics().evictions == 240-16);
  m.close_all();
  for (int i=0;i<80;++i) assert(read(p/std::to_string(i)) == "0\n1\n2\n");
}
static void emfile_retry(const fs::path &p) {
  low_limit(); auto &m = TraceFileManager::instance();
  seed(p/"new", "stale must disappear\n");
  for (int i=0;i<4;++i) m.write_line((p/std::to_string(i)).string(), "keep\n");
  std::vector<int> outside;
  for (;;) { int fd=open("/dev/null",O_RDONLY); if (fd<0) break; outside.push_back(fd); }
  assert(errno == EMFILE);
  m.write_line((p/"new").string(), "fresh\n");
  assert(m.statistics().evictions == 1);
  assert(m.statistics().open_files == 4);
  for (int fd:outside) close(fd);
  m.close_all(); assert(read(p/"new") == "fresh\n");
  for (int i=0;i<4;++i) assert(read(p/std::to_string(i)) == "keep\n");
}
static void first_failed_open(const fs::path &p) {
  capacity("2"); auto &m = TraceFileManager::instance();
  seed(p/"b", "stale\n"); m.write_line((p/"a").string(), "first\n");
  fault=OpenEmfileOnce; m.write_line((p/"b").string(), "new\n");
  assert(m.statistics().evictions == 1); m.close_all();
  assert(read(p/"b") == "new\n");
}
static void periodic_flush(const fs::path &p) {
  auto &m=TraceFileManager::instance();
  for (int i=0;i<4095;++i) m.write_line((p/"a").string(), "x");
  const auto before=flushes;
  m.write_line((p/"a").string(), "x"); assert(flushes==before+1);
  assert(read(p/"a").size()==4096);m.close_all();
}
static void expect_case(const fs::path &root, const std::string &name,
                        const std::function<void(const fs::path &)> &body,
                        const std::string &error={}) {
  const auto path=root/name; fs::create_directories(path);
  int pipefd[2]; assert(pipe(pipefd)==0);
  pid_t child=fork(); assert(child>=0);
  if (child==0) {
    close(pipefd[0]); assert(dup2(pipefd[1],STDERR_FILENO)>=0); close(pipefd[1]);
    struct rlimit no_core={0,0}; assert(setrlimit(RLIMIT_CORE,&no_core)==0);
    unsetenv("ACCELSIM_TRACE_MAX_OPEN_FILES");
    body(path); std::exit(0);
  }
  close(pipefd[1]);std::string stderr_text;char buffer[1024];ssize_t n;
  while ((n=::read(pipefd[0],buffer,sizeof(buffer)))>0) stderr_text.append(buffer,n);
  close(pipefd[0]); int status=0;assert(waitpid(child,&status,0)==child);
  const bool ok=error.empty() ? WIFEXITED(status)&&WEXITSTATUS(status)==0
    : WIFEXITED(status)&&WEXITSTATUS(status)==EXIT_FAILURE&&stderr_text.find(error)!=std::string::npos;
  if (!ok) {std::cerr<<"FAIL "<<name<<" status="<<status<<" stderr="<<stderr_text<<"\n";std::exit(1);}
  std::cout<<"PASS "<<name<<"\n"<<std::flush;
}
int main() {
  char temp[]="/tmp/trace-manager-test-XXXXXX";char *dir=mkdtemp(temp);assert(dir);fs::path root(dir);
  expect_case(root,"lru_eviction_append_order",eviction);
  expect_case(root,"partial_write_order_cap1",partial_writes);
  expect_case(root,"kernel_close_late_completion",kernel_close);
  expect_case(root,"capacity_under_low_rlimit",low_capacity);
  expect_case(root,"emfile_external_descriptors",emfile_retry);
  expect_case(root,"first_failed_open_truncates",first_failed_open);
  expect_case(root,"periodic_flush",periodic_flush);
  expect_case(root,"two_translation_units",[](const fs::path &){assert(other_translation_unit_manager()==&TraceFileManager::instance());});
  expect_case(root,"infinite_rlimit",[](const fs::path &){fault=InfiniteLimit;assert(TraceFileManager::instance().statistics().capacity==4096);fault=None;});
  expect_case(root,"override_clamped",[](const fs::path &){low_limit();capacity("4096");assert(TraceFileManager::instance().statistics().capacity==16);});
  for (const std::string value:{"-1","+2","0"," 2","2x","","184467440737095516160"})
    expect_case(root,"bad_capacity_"+std::to_string(value.size())+value, [value](const fs::path &){capacity(value.c_str());TraceFileManager::instance();}, "invalid capacity failed for ACCELSIM_TRACE_MAX_OPEN_FILES");
  expect_case(root,"open_missing_parent",[](const fs::path &p){TraceFileManager::instance().write_line((p/"absent"/"a").string(),"x");},"open failed for");
  expect_case(root,"open_denied",[](const fs::path &p){fault=OpenDenied;TraceFileManager::instance().write_line((p/"a").string(),"x");},"Permission denied");
  expect_case(root,"write_failure",[](const fs::path &p){fault=Write;TraceFileManager::instance().write_line((p/"a").string(),"x");},"write failed for");
  expect_case(root,"formatted_write_failure",[](const fs::path &p){fault=FormattedWrite;TraceFileManager::instance().writef((p/"a").string(),"%s","x");},"write failed for");
  expect_case(root,"flush_failure",[](const fs::path &p){auto &m=TraceFileManager::instance();m.write_line((p/"a").string(),"x");fault=Flush;m.flush_all();},"flush failed for");
  expect_case(root,"close_failure",[](const fs::path &p){auto &m=TraceFileManager::instance();m.write_line((p/"a").string(),"x");fault=Close;m.close_all();},"close failed for");
  expect_case(root,"buffer_failure",[](const fs::path &p){fault=Buffer;TraceFileManager::instance().write_line((p/"a").string(),"x");},"setvbuf failed for");
  expect_case(root,"limit_failure",[](const fs::path &){fault=Limit;TraceFileManager::instance();},"getrlimit failed for RLIMIT_NOFILE");
  expect_case(root,"real_dev_full",[](const fs::path &){auto &m=TraceFileManager::instance();m.write_line("/dev/full","buffered\n");m.flush_all();},"No space left on device");
  expect_case(root,"eviction_flush_failure",[](const fs::path &p){capacity("1");auto &m=TraceFileManager::instance();m.write_line((p/"a").string(),"x");fault=Flush;m.write_line((p/"b").string(),"y");},"flush failed for");
  expect_case(root,"emfile_without_evictable_handle",[](const fs::path &p){low_limit();while(open("/dev/null",O_RDONLY)>=0){}TraceFileManager::instance().write_line((p/"a").string(),"x");},"Too many open files");
  fs::remove_all(root);std::cout<<"All trace file manager tests passed\n";
}
#endif
