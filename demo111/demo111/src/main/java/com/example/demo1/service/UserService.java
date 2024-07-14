package com.example.demo1.service;

import com.example.demo1.entity.User;
import com.example.demo1.mapper.UserMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class UserService {
    @Autowired
    private UserMapper userMapper;

    public User findById(int user_id){
        User user=userMapper.findByUserID(user_id);
        return user;
    }

    public void register(int user_id,String password){
        userMapper.add(user_id,password);
    }

    public List<User> allUsers(){
        return userMapper.allUser();
    }

    public void delete(String user_id){
        userMapper.delete(user_id);
    }

    public List<User> allBlack(){
        return userMapper.selectBlack();
    }

    public List<User> allWhite(){
        return userMapper.selectWhite();
    }

    public void setBlack(int state,int user_id){
        userMapper.setBlack(state,user_id);
    }
    public void setWhite(int state,int user_id){
        userMapper.setWhite(state,user_id);
    }
}
