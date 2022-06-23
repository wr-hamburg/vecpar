#ifndef VECPAR_TIMELOGGER_HPP
#define VECPAR_TIMELOGGER_HPP

#include <string>
#include <iostream>
#include <fstream>

template <typename Arg, typename... Args>
void doPrint(std::ostream& out, Arg&& arg, Args&&... args) {
    out << std::forward<Arg>(arg);
    ((out << ',' << std::forward<Args>(args)), ...);
}

template <typename... Args>
void write_to_csv(std::string filename, Args... args) {
    std::ofstream file;
    file.open(filename, std::ios_base::out | std::ios_base::app);
    doPrint(file, args...);
    file << std::endl;
    file.close();
}

#endif //VECPAR_TIMELOGGER_HPP
