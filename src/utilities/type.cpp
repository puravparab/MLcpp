#include <iostream>

bool is_type_int(std::string value){
  try{
    // Attempt to convert to int
    std::stoi(value);
    return true;
  } catch (const std::invalid_argument&){
    return false;
  }
}
bool is_type_double(std::string value){
  try{
    // Attempt to convert to int
    std::stod(value);
    return true;
  } catch (const std::invalid_argument&){
    return false;
  }
}