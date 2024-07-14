package com.example.demo1.entity;


import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@AllArgsConstructor
@Schema(description= "返回结果信息")
@Data
public class Result<T> {
    @Schema(description= "业务状态码  0-成功  1-失败", required = true)
    private Integer code;//业务状态码  0-成功  1-失败

    @Schema(description= "提示信息")
    private String message;//提示信息

    @Schema(description= "响应数据")
    private T data;//响应数据

    //快速返回操作成功响应结果(带响应数据)
    public static <E> Result<E> success(E data) {
        return new Result<>(0, "操作成功", data);
    }

    //快速返回操作成功响应结果
    public static Result success() {
        return new Result(0, "操作成功", null);
    }

    public static Result error(String message) {
        return new Result(1, message, null);
    }
}