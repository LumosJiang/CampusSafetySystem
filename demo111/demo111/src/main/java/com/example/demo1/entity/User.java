package com.example.demo1.entity;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description= "用户信息")
public class User {
    @Schema(description = "用户id", required = true)
    private String user_id;

    @Schema(description = "名字")
    private String name;

    @Schema(description = "密码", required = true)
    private String password;

    @Schema(description = "邮箱")
    private String email;

    @Schema(description = "权限")
    private int authority;

}
