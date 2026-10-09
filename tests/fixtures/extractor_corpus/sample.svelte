<script lang="ts">
  export let name: string;

  function greet() {
    return `Hello, ${name}`;
  }
</script>

<h1>{greet()}</h1>

<style>
  h1 {
    color: tomato;
  }
</style>
