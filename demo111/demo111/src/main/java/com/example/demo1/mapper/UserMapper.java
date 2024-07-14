package com.example.demo1.mapper;

import com.example.demo1.entity.User;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface UserMapper {
    //所有用户
    @Select("select * from user")
    List<User> allUser();

    //根据用户id查询
    @Select("select * from user where user_id=#{user_id}")
    User findByUserID(int user_id);

    //新增用户
    @Insert("insert into user(user_id,password)" +
            " values(#{user_id},#{password})")
    void add(int user_id, String password);

    @Delete("delete from user where user_id=#{user_id}")
    void delete(String user_id);

    @Select("select * from user where isBlack = 1")
    List<User> selectBlack();

    @Select("select * from user where isWhite = 1")
    List<User> selectWhite();

    @Update("update user set isBlack=#{state} where user_id=#{user_id}")
    void setBlack(int state,int user_id);

    @Update("update user set isWhite=#{state} where user_id=#{user_id}")
    void setWhite(int state,int user_id);

}
