package com.example.demo1.controller;

import com.example.demo1.entity.Result;
import com.example.demo1.entity.User;
import com.example.demo1.service.UserService;
import com.example.demo1.utils.JwtUtil;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author: zyx
 * @datetime: 2024/7/10
 * @desc:
 */
@Tag(name = "用户控制器", description = "描述：管理用户信息")
@RestController
@RequestMapping("/user")
public class UserController {
    @Autowired
    private UserService userService;

    @Operation(summary = "注册用户")
    @PostMapping("/register")
    public Result register(int user_id, String password) {
        User user = userService.findById(user_id);
        if (user == null) {
            userService.register(user_id, password);
            return Result.success();
        } else {
            return Result.error("用户id已存在");
        }
    }

    @Operation(summary = "登录")
    @PostMapping("/login")
    public Result login(int user_id, String password) {
        User user = userService.findById(user_id);
        if (user == null) {
            return Result.error("用户不存在");
        }
        if (password.equals(user.getPassword())) {
            Map<String, Object> claims = new HashMap<>();
            claims.put("id", user.getUser_id());
            claims.put("name", user.getName());
            String token = JwtUtil.genToken(claims);
            return Result.success(token);
        }
        return Result.error("密码错误");
    }

    @Operation(summary = "所有用户")
    @GetMapping("/all")
    public Result allUsers() {
        List<User> userList = userService.allUsers();
        return Result.success(userList);
    }

    @Operation(summary = "删除用户")
    @PostMapping("/delete")
    public Result delete(@RequestBody User user) {
        String user_id = user.getUser_id();
        userService.delete(user_id);
        return Result.success();
    }

    @Operation(summary = "用户详细信息")
    @GetMapping("/detail")
    public Result getDetail(int user_id) {
        return Result.success(userService.findById(user_id));
    }

}
